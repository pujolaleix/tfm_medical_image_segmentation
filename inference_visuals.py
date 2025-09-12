# visualize_gt_dt.py
# ------------------------------------------------------------
# Visualize GT vs DT bboxes and masks with *non-overlapping* labels:
#   - GT (green):     TOP-LEFT of mask/box
#   - DT (red):       BOTTOM-RIGHT of mask/box
# Saves one PNG per sampled image.
# Masks are drawn as faint fill + crisp contour, aligned pixel-perfectly.
# Labels show CATEGORY NAMES (built from GT categories if id2label not provided).
# ------------------------------------------------------------

import json, random, os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as maskUtils
    _HAVE_COCO = True
except Exception:
    _HAVE_COCO = False


# ----------------------- Helpers -----------------------

def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def _decode_segm(segm, h, w):
    """Decode COCO RLE/polygon segmentation into a binary mask."""
    assert _HAVE_COCO, "pycocotools is required for mask decoding."
    if isinstance(segm, dict) and "counts" in segm:
        rle = segm
    else:
        rle = maskUtils.frPyObjects(segm, h, w)
        if isinstance(rle, list):
            rle = maskUtils.merge(rle)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)

def _load_image(imgs_index, images_dir: Path, image_id: int) -> Tuple[Image.Image, str, int, int]:
    meta = imgs_index[image_id]
    fn = meta["file_name"]
    img_path = Path(fn)
    if not img_path.is_file():
        img_path = images_dir / fn
    if not img_path.is_file():
        img_path = images_dir / os.path.basename(fn)
    im = Image.open(str(img_path)).convert("RGB")
    return im, str(img_path), int(meta["width"]), int(meta["height"])

def _draw_box(ax, xyxy, color, lw=2):
    x1, y1, x2, y2 = xyxy
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=lw, edgecolor=color)
    ax.add_patch(rect)

def _label_box(ax, xyxy, text: str, color, position="top-left"):
    x1, y1, x2, y2 = xyxy
    if position == "bottom-right":
        ax.text(
            x2, y2 + 3, text, fontsize=9, color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5),
            ha="right", va="top"
        )
    else:
        ax.text(
            x1, max(0, y1 - 3), text, fontsize=9, color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5),
            ha="left", va="bottom"
        )

def _draw_mask(ax, mask: np.ndarray, color=(1, 0, 0), lw=2,
               alpha_fill=0.20, alpha_contour=1.0):
    """
    Draw both a faint transparent fill + contour for a binary mask,
    aligned exactly with the image pixels.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape

    # ---- transparent fill (RGBA) ----
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = (mask > 0).astype(np.float32) * alpha_fill
    # Use the same extent as the base image and disable interpolation for crisp edges
    ax.imshow(overlay, extent=(0, w, h, 0), origin="upper", interpolation="none", zorder=2)

    # ---- contour in the same coordinate system ----
    # Provide explicit x/y coords so contour uses (0..w, 0..h) likewise
    xs = np.arange(w)
    ys = np.arange(h)
    ax.contour(xs, ys, mask, levels=[0.5], colors=[color],
               linewidths=lw, alpha=alpha_contour, zorder=3)

def _label_mask(ax, mask: np.ndarray, text: str, color, position="top-left"):
    m = (mask > 0)
    if not m.any():
        return
    ys, xs = np.where(m)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

    if position == "bottom-right":
        ax.text(
            x2, y2 + 3, text, fontsize=9, color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5),
            ha="right", va="top"
        )
    else:
        ax.text(
            x1, max(0, y1 - 3), text, fontsize=9, color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5),
            ha="left", va="bottom"
        )


# ----------------------- Main API -----------------------

def visualize_gt_dt_samples(
    gt_json_path: str,
    dt_bbox_json_path: Optional[str],
    dt_segm_json_path: Optional[str],
    images_dir: str,
    id2label: Optional[Dict[int, str]] = None,
    out_dir: str = "viz_out",
    num_images: int = 12,
    score_thresh: float = 0.0,
    max_dt_per_image: int = 20,
    draw_bboxes: bool = True,
    draw_masks: bool = True,
    seed: int = 0,
    only_images_with_dets: bool = False,
    # mask drawing knobs
    mask_fill_alpha: float = 0.20,
    mask_contour_alpha: float = 1.00,
    mask_contour_lw: int = 2,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    # --- Load JSONs
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    dt_bbox, dt_segm = [], []
    if dt_bbox_json_path and Path(dt_bbox_json_path).is_file():
        dt_bbox = json.load(open(dt_bbox_json_path, "r", encoding="utf-8"))
    if dt_segm_json_path and Path(dt_segm_json_path).is_file():
        dt_segm = json.load(open(dt_segm_json_path, "r", encoding="utf-8"))

    # --- id2label (auto-build if not given)
    if not id2label:
        id2label = {int(c["id"]): c.get("name", str(c["id"])) for c in gt.get("categories", [])}
    
    print(id2label)

    # --- Index GT
    imgs_index = {int(im["id"]): im for im in gt["images"]}
    gt_by_img = {}
    for ann in gt["annotations"]:
        gt_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    # --- Index DT (bboxes)
    dtb_by_img = {}
    for d in dt_bbox:
        if float(d.get("score", 0.0)) < score_thresh:
            continue
        dtb_by_img.setdefault(int(d["image_id"]), []).append(d)
    for img_id, lst in dtb_by_img.items():
        dtb_by_img[img_id] = sorted(lst, key=lambda x: -float(x.get("score", 0.0)))[:max_dt_per_image]

    # --- Index DT (masks)
    dts_by_img = {}
    for d in dt_segm:
        if float(d.get("score", 0.0)) < score_thresh:
            continue
        dts_by_img.setdefault(int(d["image_id"]), []).append(d)
    for img_id, lst in dts_by_img.items():
        dts_by_img[img_id] = sorted(lst, key=lambda x: -float(x.get("score", 0.0)))[:max_dt_per_image]

    # --- Pick images
    rng = random.Random(seed)
    all_img_ids = list(imgs_index.keys())
    if only_images_with_dets:
        all_img_ids = [i for i in all_img_ids if (i in dtb_by_img) or (i in dts_by_img)]
    rng.shuffle(all_img_ids)
    sample_ids = all_img_ids[:num_images]

    # --- Draw
    for img_id in sample_ids:
        im, img_path, W, H = _load_image(imgs_index, images_dir, img_id)
        fig, ax = plt.subplots(figsize=(8, 8))

        # IMPORTANT: show base image in the same (0..W, 0..H) extent and invert Y
        ax.imshow(im, extent=(0, W, H, 0), origin="upper", interpolation="nearest", zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # y downwards (top-left origin)
        ax.axis("off")
        #ax.set_title(f"img_id={img_id} | {os.path.basename(img_path)}")

        # ---------------- GT (green) ----------------
        for ann in gt_by_img.get(img_id, []):
            cid = int(ann["category_id"])
            cname = id2label.get(cid, f"unknown({cid})")

            # masks
            if draw_masks and _HAVE_COCO and "segmentation" in ann:
                try:
                    m = _decode_segm(ann["segmentation"], H, W)
                    _draw_mask(ax, m, color=(0, 1, 0),
                               lw=mask_contour_lw,
                               alpha_fill=mask_fill_alpha,
                               alpha_contour=mask_contour_alpha)
                    _label_mask(ax, m, f"GT: {cname}", color=(0, 1, 0), position="top-left")
                except Exception:
                    pass

            # boxes
            if draw_bboxes and "bbox" in ann:
                xyxy = _xywh_to_xyxy(ann["bbox"])
                _draw_box(ax, xyxy, color=(0, 1, 0), lw=2)
                _label_box(ax, xyxy, f"GT: {cname}", color=(0, 1, 0), position="top-left")

        # ---------------- DT (red) ----------------
        if draw_bboxes:
            for d in dtb_by_img.get(img_id, []):
                cid, s = int(d["category_id"]), float(d.get("score", 0.0))
                cname = id2label.get(cid, f"unknown({cid})")
                xyxy = _xywh_to_xyxy(d["bbox"])
                _draw_box(ax, xyxy, color=(1, 0, 0), lw=2)
                _label_box(ax, xyxy, f"DT: {cname} {s:.2f}", color=(1, 0, 0), position="bottom-right")

        if draw_masks and _HAVE_COCO:
            for d in dts_by_img.get(img_id, []):
                cid, s = int(d["category_id"]), float(d.get("score", 0.0))
                cname = id2label.get(cid, f"unknown({cid})")
                try:
                    m = _decode_segm(d["segmentation"], H, W)
                    _draw_mask(ax, m, color=(1, 0, 0),
                               lw=mask_contour_lw,
                               alpha_fill=mask_fill_alpha,
                               alpha_contour=mask_contour_alpha)
                    _label_mask(ax, m, f"DT: {cname} {s:.2f}", color=(1, 0, 0), position="bottom-right")
                except Exception:
                    pass

        out_path = out_dir / f"viz_img_{img_id}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[viz] saved {out_path}")


# ----------------------- Example -----------------------
# if __name__ == "__main__":
#     visualize_gt_dt_samples(
#         gt_json_path="path/to/instances_val.json",
#         dt_bbox_json_path="path/to/detections_boxes.json",
#         dt_segm_json_path="path/to/detections_masks.json",
#         images_dir="path/to/images",
#         id2label={1: "benign", 2: "malignant", 3: "potentially malignant"},
#         out_dir="viz_out",
#         num_images=12,
#         score_thresh=0.3,
#         max_dt_per_image=20,
#         draw_bboxes=True,
#         draw_masks=True,
#         seed=0,
#         only_images_with_dets=False,
#         mask_fill_alpha=0.10,      # tweak if you want
#         mask_contour_alpha=1.00,
#         mask_contour_lw=2,
#     )
