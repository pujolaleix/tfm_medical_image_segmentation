import json
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple
import torch
from tqdm.auto import tqdm
import torch.utils.data as tud
import numpy as np
from pycocotools import mask as mask_utils
from contextlib import contextmanager


def _shoelace_area(poly: List[List[float]]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

def _poly_to_bbox_xywh(poly: List[List[float]]) -> List[float]:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [float(x0), float(y0), float(max(0.0, x1-x0)), float(max(0.0, y1-y0))]

def _canon(name: str) -> str:
    # normalize to basename + lower-case for robust matching
    return Path(str(name)).name.lower()



def write_coco_gt_from_subset_structured(
    subset: List[Dict[str, Any]],          # <-- structured split (train_s/val_s/test_s)
    label2id: Dict[str, int],
    out_json_path: str,
    supercategory: str = "lesion"
) -> Tuple[str, Dict[str, int]]:
    """
    Create COCO GT JSON from a *structured* split.
    Uses per-ROI 'label' and 'points'; writes bbox (xywh) and segmentation (polygon).
    Returns (out_json_path, label2id).
    """
    # categories
    categories = [
        {"id": cid, "name": name, "supercategory": supercategory}
        for name, cid in sorted(label2id.items(), key=lambda kv: kv[1])
    ]

    # images & annotations
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    file2imgid: Dict[str, int] = {}
    next_img_id = 1
    ann_id = 1

    for item in subset:
        fname = item["image_id"]
        W = int(item.get("width") or 0)
        H = int(item.get("height") or 0)

        if fname not in file2imgid:
            file2imgid[fname] = next_img_id
            images.append({"id": next_img_id, "file_name": fname, "width": W, "height": H})
            next_img_id += 1
        img_id = file2imgid[fname]

        for reg in item.get("regions", []):
            # ROI label
            name = reg.get("label") or reg.get("region_label")
            if not name:
                continue
            cid = label2id.get(name)
            if cid is None:
                continue

            # polygon points (floats)
            pts = reg.get("points", [])
            if not pts or len(pts) < 3:
                continue

            # bbox xywh (prefer stored bbox if present)
            if "bbox_xyxy" in reg and reg["bbox_xyxy"]:
                x0, y0, x1, y1 = reg["bbox_xyxy"]
                bbox = [float(x0), float(y0), float(max(0.0, x1-x0)), float(max(0.0, y1-y0))]
            else:
                bbox = _poly_to_bbox_xywh(pts)

            # area from polygon (shoelace)
            area = _shoelace_area(pts)

            # segmentation as one polygon (flattened)
            seg = []
            flat = []
            for x, y in pts:
                flat.append(float(x)); flat.append(float(y))
            seg = [flat]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cid),
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "segmentation": seg
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_p = Path(out_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(coco))

    return str(out_p), label2id




@torch.inference_mode()  # faster eval; no autograd state
def write_coco_dt_from_loader(
    model,
    loader,
    device,
    file2imgid: dict,
    out_bbox_json_path: str,
    out_segm_json_path: str = None,   # optional; if provided and segm_eval=True, write segm
    score_thresh: float = 0.0,
    mask_thresh: float = 0.0,
    use_amp: bool = True,
    include_bbox_in_segm: bool = True,
    segm_eval: bool = False,
    max_dets_per_image: int = 100,    # speed cap
    epoch=0,
    total_epochs=60,
):
    """
    Runs inference once and writes:
      - COCO bbox detections to `out_bbox_json_path`
      - COCO segm detections to `out_segm_json_path` (if segm_eval=True and masks available)
    Returns: dict with written paths, counts, and #missed filename keys.
    """

    def _canon(s):
        return str(s).replace("\\", "/").lower()

    def _get_img_hw_from_tgt(tgt, fallback_hw):
        """
        Try multiple common fields to get the original image (H, W).
        Fallback to the provided value if not found.
        """
        # 1) tuple/list forms
        for key in ("orig_size", "size", "image_size"):
            v = tgt.get(key, None)
            if v is not None:
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    return int(v[0]), int(v[1])
                if isinstance(v, dict) and "height" in v and "width" in v:
                    return int(v["height"]), int(v["width"])
        # 2) separate keys
        h = tgt.get("height", None)
        w = tgt.get("width", None)
        if h is not None and w is not None:
            return int(h), int(w)
        # fallback to the tensor size (likely resized input; not ideal but better than crashing)
        return int(fallback_hw[0]), int(fallback_hw[1])

    def _paste_instance_mask_to_canvas(mask_2d_float_t, box_xyxy, img_h, img_w, thresh=0.5):
        """
        mask_2d_float_t: torch tensor [Hmask, Wmask] with values in [0,1] (probabilities/logits already sigmoided).
        box_xyxy: torch tensor/list [x1, y1, x2, y2] in ORIGINAL image coords.
        Returns a np.uint8 canvas of shape (img_h, img_w) with {0,1}.
        """
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        # clamp + convert to ints
        x0 = max(0, int(np.floor(x1)))
        y0 = max(0, int(np.floor(y1)))
        x1i = min(img_w, int(np.ceil(x2)))
        y1i = min(img_h, int(np.ceil(y2)))
        if x1i <= x0 or y1i <= y0:
            return np.zeros((img_h, img_w), dtype=np.uint8)

        box_w = x1i - x0
        box_h = y1i - y0

        # Resize mask to box size (bilinear on GPU if available)
        m = mask_2d_float_t.unsqueeze(0).unsqueeze(0)  # [1,1,Hm,Wm]
        m_up = torch.nn.functional.interpolate(m, size=(box_h, box_w), mode="bilinear", align_corners=False).squeeze()  # [box_h, box_w]
        bin_patch = (m_up >= thresh).to(torch.uint8).cpu().numpy()

        canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        canvas[y0:y1i, x0:x1i] = bin_patch
        return canvas

    def _encode_rle(mask_np, mask_utils):
        rle = mask_utils.encode(np.asfortranarray(mask_np))  # returns dict with bytes counts
        rle["counts"] = rle["counts"].decode("ascii")
        return rle

    model.eval()
    dt_bbox, dt_segm = [], []
    missed = 0

    # normalize mapping once
    f2i = { _canon(k): int(v) for k, v in file2imgid.items() }

    # import mask utils only if we might encode segm
    mask_utils_mod = None
    if segm_eval and out_segm_json_path is not None:
        try:
            from pycocotools import mask as mask_utils_mod  # CPU-only
        except Exception:
            mask_utils_mod = None
            print("[write_coco_dt_from_loader] pycocotools not available → segm JSON skipped.")

    progress_bar = tqdm(loader, desc=f"Inference detections {epoch+1}/{total_epochs}", leave=False)
    for images, targets in progress_bar:
        images = [img.to(device, non_blocking=True) for img in images]
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)

        for img_tensor, pred, tgt in zip(images, outputs, targets):
            # ----- image id mapping -----
            fname = tgt.get("image_key", tgt.get("image_id"))
            key = _canon(fname)
            image_id = f2i.get(key)
            if image_id is None:
                missed += 1
                continue
            image_id = int(image_id)

            # ----- get original image size (H, W) -----
            # fallback uses current tensor size (which might be resized)
            fallback_hw = img_tensor.shape[-2:]
            img_h, img_w = _get_img_hw_from_tgt(tgt, fallback_hw)

            boxes  = pred.get("boxes")
            labels = pred.get("labels")
            scores = pred.get("scores")
            masks  = pred.get("masks")  # expected [N,1,Hm, Wm] or [N,Hm,Wm]

            if boxes is None or labels is None or scores is None or boxes.numel() == 0:
                continue

            # select indices once: threshold + top-K
            idx = torch.arange(scores.numel(), device=scores.device)
            if score_thresh > 0.0:
                idx = idx[scores >= score_thresh]
            if idx.numel() == 0:
                continue
            if max_dets_per_image and idx.numel() > max_dets_per_image:
                topk = torch.topk(scores[idx], k=max_dets_per_image).indices
                idx = idx[topk]

            boxes  = boxes[idx]
            labels = labels[idx]
            scores = scores[idx]

            # ---- BBOX (force pixel-space xywh, per image) ----
            if out_bbox_json_path is not None:
                # boxes is Nx4, model returns xyxy in *image pixel coords* normally.
                # But if something upstream produced normalized coords, auto-upscale.

                x1, y1, x2, y2 = boxes.unbind(1)  # tensors

                # Detect "normalized-looking" coords for this image (defensive)
                # If the largest coordinate is <= ~1, treat as normalized [0,1] and scale by (img_w, img_h).
                # (Use .amax() to avoid materializing a Python float for each element.)
                max_coord = torch.max(torch.stack([x1, y1, x2, y2])).item()
                if max_coord <= 1.5:  # small tolerance
                    x1 = x1 * img_w
                    x2 = x2 * img_w
                    y1 = y1 * img_h
                    y2 = y2 * img_h

                # clamp to image bounds
                x1 = x1.clamp(min=0, max=img_w)
                y1 = y1.clamp(min=0, max=img_h)
                x2 = x2.clamp(min=0, max=img_w)
                y2 = y2.clamp(min=0, max=img_h)

                # convert xyxy -> xywh (COCO)
                w = (x2 - x1).clamp(min=0)
                h = (y2 - y1).clamp(min=0)

                dt_bbox.extend(
                    {
                        "image_id": image_id,
                        "category_id": int(c),
                        "bbox": [float(bx), float(by), float(bw), float(bh)],  # PIXELS, xywh
                        "score": float(s),
                    }
                    for bx, by, bw, bh, c, s in zip(
                        x1.tolist(), y1.tolist(), w.tolist(), h.tolist(),
                        labels.tolist(), scores.tolist()
                    )
                )

            # ---- SEGM (RLE encode per instance, in ORIGINAL image size) ----
            if segm_eval and out_segm_json_path is not None and (mask_utils_mod is not None):
                if (masks is not None) and masks.numel() > 0:
                    msel = masks[idx]
                    if msel.ndim == 4 and msel.shape[1] == 1:
                        msel = msel.squeeze(1)  # [N, Hm, Wm]
                    elif msel.ndim != 3:
                        # Unexpected shape; skip safely
                        pass

                    # Heuristic: if mask is "small" (<= 64), assume instance mask in box coords; else, treat as full-image mask.
                    # Either way, produce a canvas of (img_h, img_w) before encoding.
                    for i in range(msel.shape[0]):
                        # defensive: some models output logits; if so, apply sigmoid
                        m_i = msel[i]
                        if m_i.dtype.is_floating_point:
                            # ensure probabilities
                            m_i = m_i.sigmoid() if (m_i.min() < 0 or m_i.max() > 1) else m_i
                        else:
                            m_i = m_i.float()

                        # Decide path
                        Hm, Wm = m_i.shape[-2], m_i.shape[-1]
                        if (Hm <= 64 and Wm <= 64):
                            # Instance mask (e.g., 28x28) → paste into bbox on full canvas
                            canvas = _paste_instance_mask_to_canvas(
                                m_i, boxes[i], img_h, img_w, thresh=mask_thresh
                            )
                        else:
                            # Full-image-ish mask (but may be at network/input size) → upsample to original image size
                            m_up = torch.nn.functional.interpolate(
                                m_i.unsqueeze(0).unsqueeze(0),  # [1,1,Hm,Wm]
                                size=(img_h, img_w),
                                mode="bilinear",
                                align_corners=False
                            ).squeeze(0).squeeze(0)
                            canvas = (m_up >= mask_thresh).to(torch.uint8).cpu().numpy()

                        # Encode one-by-one to avoid huge (H,W,N) allocations
                        try:
                            rle = _encode_rle(canvas, mask_utils_mod)
                        except Exception as e:
                            # As a last resort, write an empty mask rather than crash
                            canvas[:] = 0
                            rle = _encode_rle(canvas, mask_utils_mod)

                        rec = {
                            "image_id": image_id,
                            "category_id": int(labels[i].item()),
                            "segmentation": rle,
                            "score": float(scores[i].item()),
                        }
                        if include_bbox_in_segm:
                            x1_, y1_, x2_, y2_ = [float(v) for v in boxes[i]]
                            w_ = max(0.0, x2_ - x1_); h_ = max(0.0, y2_ - y1_)
                            rec["bbox"] = [float(x1_), float(y1_), float(w_), float(h_)]
                        dt_segm.append(rec)

    # ---- write files ----
    out = {"missed": int(missed)}
    if out_bbox_json_path is not None:
        Path(out_bbox_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_bbox_json_path, "w", encoding="utf-8") as f:
            json.dump(dt_bbox, f)
        out["bbox_path"] = str(out_bbox_json_path)
        out["num_dets_bbox"] = len(dt_bbox)

    if segm_eval and out_segm_json_path is not None and ("mask_utils_mod" in locals() and mask_utils_mod is not None):
        Path(out_segm_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_segm_json_path, "w", encoding="utf-8") as f:
            json.dump(dt_segm, f)
        out["segm_path"] = str(out_segm_json_path)
        out["num_dets_segm"] = len(dt_segm)
        print('dt saved')
    elif segm_eval and out_segm_json_path is not None:
        out["segm_path"] = None
        out["num_dets_segm"] = 0
        print("[write_coco_dt_from_loader] pycocotools not available → segm JSON skipped.")

    return out



def coco_stats_dict(e):
    names = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"
    ]
    return {names[i]: float(e.stats[i]) for i in range(len(names))}


