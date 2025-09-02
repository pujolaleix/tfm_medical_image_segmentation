# Imports & Dependencies
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import pandas as pd
import argparse
import numpy as np
from contextlib import contextmanager

from preparation_new import prepare_structured_dataset_df, build_label_mapping, detection_collate, patient_level_split, OralLesionDataset
from augmentations import get_transforms
from models import model_selector
from coco_io import write_coco_gt_from_subset_structured, write_coco_dt_from_loader, _canon
from metrics import run_metrics
from visuals import visualize_and_save_augmentations



class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # linear warmup
            return [base_lr * float(epoch + 1) / float(self.warmup_epochs) for base_lr in self.base_lrs]
        # cosine schedule after warmup
        progress = (epoch - self.warmup_epochs) / float(self.max_epochs - self.warmup_epochs)
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
            for base_lr in self.base_lrs
        ]
    



def train_one_epoch(model, loader, optimizer, device, class_weights=None, use_amp=True, clip_norm=1.0, epoch=0, total_epochs=25):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))
    total = 0.0

    progress_bar = tqdm(loader, desc=f"Training step {epoch+1}/{total_epochs}", leave=False)
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        # move masks/boxes/labels
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict

        scaler.scale(loss).backward()
        if clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().item())

    return total / max(1, len(loader))



@torch.no_grad()
def compute_val_loss(model, val_loader, device, use_amp: bool = True, epoch=0, total_epochs=25):
    """
    Returns (val_loss_scalar, avg_loss_dict) where avg_loss_dict has keys like:
    loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg
    """
    was_training = model.training
    model.train()
    total = 0.0
    sums = {}
    n_batches = 0

    progress_bar = tqdm(val_loader, desc=f"Validation step {epoch+1}/{total_epochs}", leave=False)
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):
            loss_dict = model(images, targets)

        # sum numeric components
        for k, v in loss_dict.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().item())

        loss = sum(loss_dict.values())
        total += float(loss.detach().item())
        n_batches += 1

    if not was_training:
        model.train()

    denom = max(1, n_batches)
    avg_loss = {k: v/denom for k,v in sums.items()}

    return total/denom, avg_loss



@contextmanager
def simple_eval_limits(
    model,
    enable=True,
    dets_per_img=20,        # FINAL detections per image
    score_thresh=0.15,      # prune weak boxes earlier
    min_size=512,           # smaller input side
    max_size=800,
    verbose=False,
):
    """
    Safe, version-agnostic eval limits. Does NOT touch RPN internals.
    Only caps final detections + image sizing + score threshold.
    """
    if not enable:
        yield
        return

    roi = model.roi_heads
    tfm = model.transform

    # save
    old = {
        "dets": getattr(roi, "detections_per_img", 100),
        "thr": getattr(roi, "score_thresh", 0.05),
        "min": list(tfm.min_size) if isinstance(tfm.min_size, (list, tuple)) else [int(tfm.min_size)],
        "max": int(tfm.max_size),
    }

    try:
        roi.detections_per_img = int(dets_per_img)
        roi.score_thresh = float(score_thresh)
        tfm.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        tfm.max_size = int(max_size)

        if verbose:
            print(f"[simple_eval_limits] dets/img: {old['dets']} -> {roi.detections_per_img}, "
                  f"score: {old['thr']} -> {roi.score_thresh}, "
                  f"min_size: {old['min']} -> {tfm.min_size}, max_size: {old['max']} -> {tfm.max_size}")

        yield
    finally:
        roi.detections_per_img = old["dets"]
        roi.score_thresh = old["thr"]
        tfm.min_size = old["min"]
        tfm.max_size = old["max"]
        if verbose:
            print("[simple_eval_limits] restored.")




def main(
    images_dir: str,
    df_roi: pd.DataFrame,
    df_meta: pd.DataFrame,
    mapping_target_label: pd.DataFrame,
    model_type: str = "cusom_roi",
    output_dir: str = "./outputs",
    label_field: str = "injury_label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augmentation_type: str = 'light',
    batch_size: int = 2,
    epochs: int = 25,
    lr: float = 0.01,
    num_workers: int = 2,
    score_thr: float = 0.05,
    mask_thr: float = 0.05,
    iou_thr: float = 0.50,
    dice_thr: float = 0.50,
    pretrained: bool = True,
    use_amp: bool = True,
    earlystop_metric: str = 'val',
    segm_every: int = 5,
):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load tables
    df_roi = df_roi.copy()
    df_meta = df_meta.copy()


    # 2) Build structured items
    structured = prepare_structured_dataset_df(df_roi, df_meta, mapping_target_label, images_dir, label_field=label_field)


    # 3) Labels mapping
    label2id, id2label = build_label_mapping(structured)
    print("Classes:", id2label)


    # 4) Split
    #ds_train, ds_val, ds_test = train_val_test_split(structured, (train_ratio, val_ratio, test_ratio), images_dir, label2id)
    train_s, val_s, test_s, (train_pat, val_pat, test_pat) = patient_level_split(structured, ratios=(train_ratio, val_ratio, test_ratio), label2id=label2id, out_dir=output_dir)
    print(f"Patient db size:  {len(train_pat) + len(val_pat) + len(test_pat)}")
    print(f"train size:  {len(train_pat)} | val size:  {len(val_pat)} | test size:  {len(test_pat)}")
    print(f"images -> train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")


    # Data Augmentation
    train_tfms = get_transforms(train=True,  strength=augmentation_type)
    val_tfms   = get_transforms(train=False)

    # 5) Dataset class
    ds_train = OralLesionDataset(train_s, label2id, transform=train_tfms)
    ds_val   = OralLesionDataset(val_s,   label2id, transform=val_tfms)
    ds_test  = OralLesionDataset(test_s,  label2id)

    visualize_and_save_augmentations(ds_train, save_dir=str(output_dir/"aug_examples"), num_samples=10)


    # 6) DataLoaders & Weighted Sampler
    labels = [label2id[s["case_label"]] for s in train_s]
    class_counts = np.bincount(labels, minlength=(1 + len(label2id)))
    inv_freq = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    sample_weights = [inv_freq[l] for l in labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=detection_collate)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=detection_collate)


    # 7) Model
    num_classes = 1 + len(label2id)  # background + classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model_selector(train_s, label2id, device, model_type, pretrained, num_classes)
    


    # 8) Optimizer & LR Scheduler & Early Stopping
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=2, max_epochs=epochs, min_lr=1e-6)

    best_metric = float("inf")
    early_stop_patience = 8
    bad_epochs = 0


    # 9) Write val COCO GT once (and keep mapping)
    gt_json_path_val, _ = write_coco_gt_from_subset_structured(val_s, label2id, str(output_dir/"coco_gt_val.json"))
    gt_json_path_test, _ = write_coco_gt_from_subset_structured(test_s, label2id, str(output_dir/"coco_gt_test.json"))
    print("Wrote val GT to", gt_json_path_val)
    print("Wrote test GT to", gt_json_path_test)

    dt_bbox_json_path = output_dir/"coco_dt_bbox_val.json"
    dt_segm_json_path = output_dir/"coco_dt_segm_val.json"

    # 10) Train + evaluate
    for epoch in range(epochs):

        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_amp=use_amp, epoch=epoch, total_epochs=epochs)
        print(f"train loss: {train_loss:.4f}")

        val_loss, val_loss_parts = compute_val_loss(model, val_loader, device, use_amp=use_amp, epoch=epoch, total_epochs=epochs)
        print(f"validation loss: {val_loss:.4f}")

        segm_flag=((epoch + 1) % segm_every == 0 or (epoch + 1) == epochs)

        # Inference on val test
        gt = json.load(open(gt_json_path_val, "r"))
        file2imgid = {_canon(im["file_name"]): im["id"] for im in gt["images"]}
        with simple_eval_limits(model, enable=segm_flag, dets_per_img=20, score_thresh=score_thr, min_size=512, max_size=800, verbose=True):
            out = write_coco_dt_from_loader(
                model, val_loader, device,
                file2imgid,
                out_bbox_json_path=str(dt_bbox_json_path),
                out_segm_json_path=str(dt_segm_json_path),
                score_thresh=score_thr, mask_thresh=mask_thr, use_amp=True,
                segm_eval=segm_flag,
                epoch=epoch,
                total_epochs=epochs,
            )

        # Evaluation & metrics
        out_metric = run_metrics(gt_json_path = gt_json_path_val,
                    dt_bbox_path = dt_bbox_json_path,
                    dt_segm_path = dt_segm_json_path,
                    out_dir = output_dir,
                    id2label = id2label,
                    epoch = epoch,
                    iou_thr = iou_thr,
                    dice_thr= dice_thr,
                    score_thr= score_thr,
                    mask_thr= mask_thr,
                    agg_acc_im= 'max',
                    segm_flag = segm_flag,
                    loss_metric = earlystop_metric,
        )


        # Early Stopping & Best Checkpoint saving
        if out_metric==None:
            cur_metric = val_loss
        else:
            cur_metric = out_metric


        if cur_metric > best_metric + 1e-4:  # small tolerance
            best_metric = cur_metric
            bad_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        #LR Scheduler
        #scheduler.step(cur_metric)     # for plateau scheduler
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"LR now: {cur_lr:.6f}")
        
    # Save final model
    torch.save(model.state_dict(), str(output_dir/"maskrcnn_final.pt"))
    print("Saved model to", output_dir/"maskrcnn_final.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Directory with images (filenames matching CSV 'image_id')")
    ap.add_argument("--csv_path", default="oralmedROI_v10.csv")
    ap.add_argument("--xlsx_path", default="oralmedDS_v10.xlsx")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--label_field", default="injury_label", choices=["injury_label", "outcome_label"])
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    main(
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        xlsx_path=args.xlsx_path,
        output_dir=args.output_dir,
        label_field=args.label_field,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp,
    )
