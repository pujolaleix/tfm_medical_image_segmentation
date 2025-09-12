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
from typing import List, Dict, Any, Tuple, Optional

from preparation_new import prepare_structured_dataset_df, build_label_mapping, detection_collate, patient_level_split, OralLesionDataset, write_metrics_txt, write_args_to_txt
from augmentations import get_transforms
from balancing import downsample_for_balancing
from models import model_selector, freeze_stem_layer1
from coco_io import write_coco_gt_from_subset_structured, write_coco_dt_from_loader, _canon
from metrics import run_metrics
from visuals import visualize_and_save_augmentations
from sanity import *



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
    mapping_target_label: Optional[pd.DataFrame],
    mode_type: str = "train",
    model_type: str = "custom_roi",
    output_dir: str = "./outputs",
    label_field: str = "injury_label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augmentation_type: str = 'light',
    downsampling_flag: bool = False,
    balance_skew: float = 0.5,
    balance_max_ratio: float = None,
    keep_one_bbox: bool = True,
    selection_mode: str = 'largest_area',
    batch_size: int = 2,
    epochs: int = 25,
    lr: float = 0.01,
    num_workers: int = 2,
    score_thr: float = 0.00,
    mask_thr: float = 0.00,
    iou_thr: float = 0.50,
    dice_thr: float = 0.50,
    img_acc_score_thr: float = 0.50,
    pretrained: bool = True,
    use_amp: bool = True,
    optimize_metric: Optional[str] = None,
    segm_every: int = 5,
    scheduler_type: str = "cosine",
    sampler_alpha: float = 0.7,
    max_class_weights: float = 2.0,
    dice_alpha: float = 0.5,
    use_bce: bool = True,
    use_focal: bool = False,
    focal_alpha: float = 0.25, 
    focal_gamma: float = 2.0, 
    label_smoothing: float = 0.05,
    dropout_alpha: float = 0.2,
):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (mode_type=='train'):
        write_args_to_txt(locals(), str(output_dir/"arguments.txt"))

    # 1) Load tables
    df_roi = df_roi.copy()
    df_meta = df_meta.copy()


    # 2) Build structured items
    structured = prepare_structured_dataset_df(df_roi, df_meta, mapping_target_label, images_dir, label_field=label_field, 
                                               stats_txt_path=str(output_dir/"n_bbox_dist"), keep_one_bbox=keep_one_bbox, selection_mode=selection_mode,
                                               downsampling_flag=downsampling_flag, balance_skew=balance_skew, balance_max_ratio=balance_max_ratio)
    

    # 3) Labels mapping
    label2id, id2label = build_label_mapping(structured)
    print("Classes:", id2label)
    print(label2id)


    # 4) Split
    #ds_train, ds_val, ds_test = train_val_test_split(structured, (train_ratio, val_ratio, test_ratio), images_dir, label2id)
    train_s, val_s, test_s, (train_pat, val_pat, test_pat) = patient_level_split(structured, ratios=(train_ratio, val_ratio, test_ratio), label2id=label2id, out_dir=output_dir)
    print(f"Patient db size:  {len(train_pat) + len(val_pat) + len(test_pat)}")
    print(f"train size:  {len(train_pat)} | val size:  {len(val_pat)} | test size:  {len(test_pat)}")
    print(f"images -> train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")


    # 5) Downsampling
    if downsampling_flag:
        print('Size before downsampling: ', len(train_s))
        train_s = downsample_for_balancing(train_s=train_s, skew=balance_skew, max_ratio=balance_max_ratio)
        print('Size after downsampling: ', len(train_s))



    # Data Augmentation
    train_tfms = get_transforms(train=True,  strength=augmentation_type)
    val_tfms   = get_transforms(train=False)

    # 5) Dataset class
    ds_train = OralLesionDataset(train_s, label2id, transform=train_tfms)
    ds_val   = OralLesionDataset(val_s,   label2id, transform=val_tfms)
    ds_test  = OralLesionDataset(test_s,  label2id)

    # dataset_sanity(ds_train, "train")
    # dataset_sanity(ds_val,   "val")

    visualize_and_save_augmentations(ds_train, save_dir=str(output_dir/"aug_examples"), num_samples=10)

    # 6) DataLoaders & Weighted Sampler controled by sampler_alpha
    labels = np.array([label2id[s["case_label"]] for s in train_s], dtype=np.int64)
    K = len(label2id)  # foreground classes only (exclude background)

    # bincount with background slot (index 0) and foreground 1..K
    counts = np.bincount(labels, minlength=K+1)   # shape [K+1], idx 0 unused/empty here
    fg_counts = counts[1:]                        # shape [K], classes 1..K -> 0..K-1

    alpha = sampler_alpha  # e.g., 0.5
    fg_weights = (1.0 / np.clip(fg_counts, 1, None)) ** alpha

    # normalize so mean weight ~ 1 (stabilizes training scale)
    fg_weights = fg_weights * (K / fg_weights.sum())

    # per-sample weights: note the -1 offset from label in {1..K}
    sample_weights = fg_weights[labels - 1]

    # (Optional) readable sanity print with class names
    id2label = {v: k for k, v in label2id.items()}
    print("\nClass weights (per label):")
    for i in range(K):
        label_id = i + 1   # since your foreground labels are 1..K
        print(f"{label_id}: {fg_weights[i]:.6f}")
        # Avoid Counter(sample_weights) on floats; not meaningful.

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=detection_collate)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=detection_collate)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=detection_collate)


    # 7) Model
    num_classes = 1 + len(label2id)  # background + classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model_selector(train_s, label2id, device, model_type, pretrained, num_classes, max_class_weights, 
                            dice_alpha, use_bce, use_focal, focal_alpha, focal_gamma, label_smoothing, dropout_alpha)
    

    print(f'Entering {mode_type} mode...')
    if (mode_type=="train"):
        
        # 8) Optimizer & LR Scheduler & Early Stopping
        minimize_flag = (optimize_metric is None) #meaning metric is monitored loss --> val loss
        best_metric = float("inf") if minimize_flag else -float("inf")     # maximize for any metric but monitoring loss
        tolerance = 1e-4
        early_stop_patience = 7
        bad_epochs = 0

        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-5)

        if (scheduler_type=="cosine"):
            scheduler = WarmupCosineLR(optimizer, warmup_epochs=2, max_epochs=epochs, min_lr=1e-6)
        elif (scheduler_type=="plateau"):
            mode = "min" if minimize_flag else "max"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.1, patience=3, min_lr=1e-6, verbose=True)
        else:
            print(f"Scheduler type {scheduler_type} not implemented.")


        # 9) Write val COCO GT once (and keep mapping)
        gt_json_path_val, _ = write_coco_gt_from_subset_structured(val_s, label2id, str(output_dir/"coco_gt_val.json"))
        gt_json_path_test, _ = write_coco_gt_from_subset_structured(test_s, label2id, str(output_dir/"coco_gt_test.json"))
        print("Wrote val GT to", gt_json_path_val)
        print("Wrote test GT to", gt_json_path_test)

        dt_bbox_json_path = output_dir/"coco_dt_bbox_val.json"
        dt_segm_json_path = output_dir/"coco_dt_segm_val.json"

        train_losses = {}
        val_losses = {}

        # 10) Train + evaluate
        for epoch in range(epochs):

            print(f"\n=== Epoch {epoch+1}/{epochs} ===")

            # Train one epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, device, use_amp=use_amp, epoch=epoch, total_epochs=epochs)
            train_losses[epoch] = train_loss
            print(f"train loss: {train_loss:.4f}")

            val_loss, val_loss_parts = compute_val_loss(model, val_loader, device, use_amp=use_amp, epoch=epoch, total_epochs=epochs)
            val_losses[epoch] = val_loss
            print(f"validation loss: {val_loss:.4f}")

            segm_flag = ((epoch + 1) % segm_every == 0 or (epoch + 1) == epochs)

            # Inference on val test
            gt = json.load(open(gt_json_path_val, "r"))
            file2imgid = {_canon(im["file_name"]): im["id"] for im in gt["images"]}
            with simple_eval_limits(model, enable=segm_flag, dets_per_img=20, score_thresh=score_thr, min_size=512, max_size=800, verbose=True):  
                out = write_coco_dt_from_loader(
                    model, val_loader, device,
                    file2imgid,
                    out_bbox_json_path=str(dt_bbox_json_path),
                    out_segm_json_path=str(dt_segm_json_path),
                    score_thresh=score_thr, 
                    mask_thresh=mask_thr, 
                    use_amp=True,
                    segm_eval=segm_flag,
                    epoch=epoch,
                    total_epochs=epochs,
                )
            
            print(out)

            if out['num_dets_bbox']>0:

                # Evaluation & metrics
                out_metric = run_metrics(gt_json_path = gt_json_path_val,
                            dt_bbox_path = dt_bbox_json_path,
                            dt_segm_path = dt_segm_json_path,
                            out_dir = Path(output_dir/mode_type),
                            id2label = id2label,
                            epoch = epoch,
                            score_thr=score_thr,
                            iou_thr = iou_thr,
                            dice_thr= dice_thr,
                            img_acc_score_thr= img_acc_score_thr,
                            agg_acc_im= 'max',
                            segm_flag = segm_flag,
                            loss_metric = optimize_metric,
                            mode_type = mode_type,
                )


            # Early Stopping & Best Checkpoint saving
            if optimize_metric==None:
                cur_metric = val_loss
            else:
                cur_metric = out_metric

            improved_flag = (cur_metric < best_metric - tolerance) if minimize_flag else (cur_metric > best_metric + tolerance)

            if improved_flag:
                best_metric = cur_metric
                bad_epochs = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print(f'current bad epochs: {bad_epochs}')


            #LR Scheduler
            if (scheduler_type=="plateau"):
                scheduler.step(cur_metric)
            else:
                scheduler.step()

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"LR now: {cur_lr:.6f}")
            
        # Save final model and losses
        torch.save(model.state_dict(), str(output_dir/"maskrcnn_final.pt"))
        print("Saved model to", output_dir/"maskrcnn_final.pt")

        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        write_metrics_txt(train_losses, str(output_dir/"plots/train_loss_ev.txt"))
        write_metrics_txt(val_losses, str(output_dir/"plots/val_loss_ev.txt"))


    elif (mode_type=="test"):

        ckpt_path = Path(output_dir) / "best_model.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(output_dir) / "maskrcnn_final.pt"
        print(f"[test] Loading checkpoint: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        # --- (re)write GT JSONs for val/test (source of truth for image ids)
        gt_json_path_val, _ = write_coco_gt_from_subset_structured(val_s,  label2id, str(output_dir / "coco_gt_val.json"))
        gt_json_path_test, _ = write_coco_gt_from_subset_structured(test_s, label2id, str(output_dir / "coco_gt_test.json"))
        print("GT (val): ", gt_json_path_val)
        print("GT (test):", gt_json_path_test)

        # --- run two splits the same way
        splits = [
            #("val",  val_loader,  gt_json_path_val),
            ("test", test_loader, gt_json_path_test),
        ]

        for split_name, loader_split, gt_path in splits:
            print(f"\n[test] Split = {split_name}")

            # DT output paths
            dt_bbox_json_path = Path(output_dir) / f"coco_dt_bbox_{split_name}.json"
            dt_segm_json_path = Path(output_dir) / f"coco_dt_segm_{split_name}.json"

            # file->img_id mapping from GT
            gt = json.load(open(gt_path, "r"))
            file2imgid = {_canon(im["file_name"]): im["id"] for im in gt["images"]}

            # --- inference -> write DTs
            # Important: score_thresh=0.0 (keep all), mask_thresh=0.5 (binarize masks)
            with simple_eval_limits(model, enable=True, dets_per_img=5, score_thresh=0.0, min_size=512, max_size=800, verbose=True,):
                _ = write_coco_dt_from_loader(
                    model=model,
                    loader=loader_split,
                    device=device,
                    file2imgid=file2imgid,
                    out_bbox_json_path=str(dt_bbox_json_path),
                    out_segm_json_path=str(dt_segm_json_path),
                    score_thresh=0.0,          # keep all for eval
                    mask_thresh=0.5,           # mask binarization (probs)
                    use_amp=True,
                    include_bbox_in_segm=True,
                    segm_eval=True,            # write masks too
                    max_dets_per_image=100,    # speed cap (per-image)
                    epoch=0,
                    total_epochs=1,
                )

            # --- evaluate
            metrics_out = run_metrics(
                gt_json_path=gt_path,
                dt_bbox_path=str(dt_bbox_json_path),
                dt_segm_path=str(dt_segm_json_path),
                out_dir=str(output_dir/mode_type/split_name),
                id2label=id2label,
                epoch=0,                       # just for filename stamping
                iou_thr=iou_thr,               # usually 0.50
                dice_thr=dice_thr,             # usually 0.50
                score_thr=0.0,                 # evaluation score filter (keep all)
                img_acc_score_thr=img_acc_score_thr,  # e.g., 0.05
                agg_acc_im="max",
                segm_flag=True,                # evaluate masks too
                loss_metric=None,
                mode_type=mode_type,
            )
            print(f"[test] Finished metrics for {split_name} → written to {str(output_dir/mode_type/split_name)}/")


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
