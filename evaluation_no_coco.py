import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from pycocotools import mask as mask_utils  # only needed for mask IoU
    _HAVE_MASK = True
except Exception:
    _HAVE_MASK = False


# -------------------- Utilities --------------------

def _bbox_xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def _greedy_match(iou_mat: np.ndarray, thr: float) -> List[float]:
    """
    Greedy 1-to-1 matching like COCO:
      detections are assumed sorted by score DESC (rows),
      for each detection pick unmatched GT with highest IoU >= thr.
    Returns IoUs for matched pairs (for averaging).
    """
    if iou_mat.size == 0:
        return []
    D, G = iou_mat.shape
    gt_used = np.zeros(G, dtype=bool)
    matched_ious = []
    for d in range(D):
        # best GT for this detection
        g = np.argmax(iou_mat[d])
        best_iou = iou_mat[d, g]
        if best_iou >= thr and not gt_used[g]:
            gt_used[g] = True
            matched_ious.append(float(best_iou))
    return matched_ious

def _to_rle(segm, h: int, w: int):
    """
    Normalize a COCO segmentation to RLE.
    segm can be:
      - RLE dict {"size":[h,w], "counts": ...}
      - Polygon list [[x1,y1,...], [...], ...]
    """
    if not _HAVE_MASK:
        raise RuntimeError("pycocotools is required for mask metrics.")
    if isinstance(segm, dict) and "counts" in segm:
        # ensure size is present
        rle = dict(segm)
        if "size" not in rle:
            rle["size"] = [h, w]
        return rle
    # polygons
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        return mask_utils.merge(rles)
    raise ValueError(f"Unsupported segmentation format: {type(segm)}")

def _mask_iou_matrix(dts: List[dict], gts: List[dict], h: int, w: int) -> np.ndarray:
    if not _HAVE_MASK:
        return np.zeros((len(dts), len(gts)), dtype=np.float32)
    if len(dts) == 0 or len(gts) == 0:
        return np.zeros((len(dts), len(gts)), dtype=np.float32)
    rle_d = [ _to_rle(d["segmentation"], h, w) for d in dts ]
    rle_g = [ _to_rle(g["segmentation"], h, w) for g in gts ]
    gt_crowd = [ int(g.get("iscrowd", 0)) for g in gts ]
    iou = mask_utils.iou(rle_d, rle_g, gt_crowd)  # shape [D,G]
    return np.asarray(iou, dtype=np.float32)


# -------------------- Loading GT/DT --------------------

def _index_coco_jsons(gt_json_path: str, dt_json_path: str):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(dt_json_path, "r", encoding="utf-8") as f:
        dt = json.load(f)

    # Map image_id -> (h,w)
    img_hw = {}
    for im in gt.get("images", []):
        img_hw[int(im["id"])] = (int(im.get("height", 0)), int(im.get("width", 0)))

    # GT by (img, cat)
    gts = defaultdict(list)
    for ann in gt.get("annotations", []):
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        gts[(img_id, cat_id)].append(ann)

    # DT by (img, cat), sorted by score DESC
    dts = defaultdict(list)
    for d in dt:
        img_id = int(d["image_id"])
        cat_id = int(d["category_id"])
        dts[(img_id, cat_id)].append(d)
    for k in dts:
        dts[k].sort(key=lambda x: -float(x.get("score", 0.0)))

    # List of categories in GT
    cat_ids = sorted({int(c["id"]) for c in gt.get("categories", [])})
    img_ids = sorted(img_hw.keys())
    return gt, dt, img_hw, gts, dts, cat_ids, img_ids





# -------------------- BBOX IoU & Dice (JSON-only) --------------------

def bbox_iou_dice_from_json(
    gt_json_path: str,
    dt_json_path: str,
    iou_thr: float = 0.50,
    score_thresh: float = 0.0,
    max_dets: int = 300,
) -> Tuple[float, Dict[int, float], float, Dict[int, float]]:
    """
    Returns: (meanIoU, perClassIoU, meanDice, perClassDice)
    Computed over matched bbox pairs using greedy COCO-like matching at `iou_thr`.
    """
    _, _, _, gts, dts, cat_ids, img_ids = _index_coco_jsons(gt_json_path, dt_json_path)

    per_cls_ious: Dict[int, List[float]] = defaultdict(list)

    for img_id in img_ids:
        for cat_id in cat_ids:
            gt_list = gts.get((img_id, cat_id), [])
            dt_list = [d for d in dts.get((img_id, cat_id), []) if float(d.get("score", 0.0)) >= score_thresh]
            if len(dt_list) > max_dets:
                dt_list = dt_list[:max_dets]
            if not gt_list or not dt_list:
                continue

            # Build IoU matrix (D x G)
            d_boxes = np.stack([_bbox_xywh_to_xyxy(d["bbox"]) for d in dt_list], axis=0)
            g_boxes = np.stack([_bbox_xywh_to_xyxy(g["bbox"]) for g in gt_list], axis=0)

            iou_mat = np.zeros((len(d_boxes), len(g_boxes)), dtype=np.float32)
            for i in range(len(d_boxes)):
                for j in range(len(g_boxes)):
                    iou_mat[i, j] = _iou_xyxy(d_boxes[i], g_boxes[j])

            matched = _greedy_match(iou_mat, iou_thr)
            if matched:
                per_cls_ious[cat_id].extend(matched)

    all_ious = [iou for vs in per_cls_ious.values() for iou in vs]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    per_class_iou = {cid: float(np.mean(vs)) for cid, vs in per_cls_ious.items() if vs}

    # Dice from IoU: Dice = 2*IoU / (1+IoU)
    mean_dice = float((2.0 * mean_iou) / (1.0 + mean_iou)) if mean_iou > 0 else 0.0
    per_class_dice = {cid: float((2.0 * v) / (1.0 + v)) for cid, v in per_class_iou.items()}

    return mean_iou, per_class_iou, mean_dice, per_class_dice


# -------------------- MASK IoU & Dice (JSON-only, if segm present) --------------------

def mask_iou_dice_from_json(
    gt_json_path: str,
    dt_json_path: str,
    iou_thr: float = 0.50,
    score_thresh: float = 0.0,
    max_dets: int = 300,
) -> Tuple[float, Dict[int, float], float, Dict[int, float]]:
    """
    Same as bbox version, but uses 'segmentation' to compute mask IoU/Dice.
    Requires pycocotools installed and both GT & DT to carry segmentations.
    """
    if not _HAVE_MASK:
        raise RuntimeError("pycocotools is required for mask IoU/Dice.")

    gt, _, img_hw, gts, dts, cat_ids, img_ids = _index_coco_jsons(gt_json_path, dt_json_path)

    per_cls_ious: Dict[int, List[float]] = defaultdict(list)

    for img_id in img_ids:
        h, w = img_hw[img_id]
        for cat_id in cat_ids:
            gt_list = [g for g in gts.get((img_id, cat_id), []) if "segmentation" in g]
            dt_list = [d for d in dts.get((img_id, cat_id), []) if "segmentation" in d and float(d.get("score", 0.0)) >= score_thresh]
            if len(dt_list) > max_dets:
                dt_list = dt_list[:max_dets]
            if not gt_list or not dt_list:
                continue

            iou_mat = _mask_iou_matrix(dt_list, gt_list, h, w)  # D x G
            matched = _greedy_match(iou_mat, iou_thr)
            if matched:
                per_cls_ious[cat_id].extend(matched)

    all_ious = [iou for vs in per_cls_ious.values() for iou in vs]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    per_class_iou = {cid: float(np.mean(vs)) for cid, vs in per_cls_ious.items() if vs}
    mean_dice = float((2.0 * mean_iou) / (1.0 + mean_iou)) if mean_iou > 0 else 0.0
    per_class_dice = {cid: float((2.0 * v) / (1.0 + v)) for cid, v in per_class_iou.items()}

    return mean_iou, per_class_iou, mean_dice, per_class_dice


# -------------------- Image-level classification accuracy (JSON-only) --------------------

def image_level_label_accuracy_from_json(
    gt_json_path: str,
    dt_json_path: str,
    agg: str = "max",              # "max" | "sum"
    score_thresh: float = 0.0,
    gt_mode: str = "majority",     # "majority" | "any"
    count_blank_as_wrong: bool = True,
) -> Tuple[float, Dict[int, float]]:
    """
    Build a single predicted class per image from detections and compare to GT image label.
    - agg='max': pick the class with the single highest detection score.
      agg='sum': sum scores per class, then argmax.
    - gt_mode='majority': GT label is the most frequent category in that image.
      gt_mode='any': GT set = all categories in image; prediction is correct if it's in the set.
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(dt_json_path, "r", encoding="utf-8") as f:
        dt = json.load(f)

    # GT labels per image
    labels_per_img = defaultdict(list)
    for ann in gt.get("annotations", []):
        labels_per_img[int(ann["image_id"])].append(int(ann["category_id"]))

    if gt_mode == "any":
        gt_targets = {img_id: set(labs) for img_id, labs in labels_per_img.items() if labs}
    else:
        gt_targets = {}
        for img_id, labs in labels_per_img.items():
            if not labs:
                continue
            c = Counter(labs)
            top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            gt_targets[img_id] = int(top)

    # DT per image (filtered by score)
    dts_by_img = defaultdict(list)
    for d in dt:
        s = float(d.get("score", 0.0))
        if s >= score_thresh:
            dts_by_img[int(d["image_id"])].append((int(d["category_id"]), s))

    total = 0
    correct = 0
    per_cls_tot = Counter()
    per_cls_cor = Counter()

    for img_id, gt_lab in gt_targets.items():
        scores = dts_by_img.get(img_id, [])

        if not scores:
            total += 1
            if gt_mode == "majority":
                per_cls_tot[int(gt_lab)] += 1
            if not count_blank_as_wrong:
                correct += 1
                if gt_mode == "majority":
                    per_cls_cor[int(gt_lab)] += 1
            continue

        if agg == "sum":
            acc = defaultdict(float)
            for cid, s in scores: acc[cid] += s
            pred = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            pred = sorted(scores, key=lambda kv: (-kv[1], kv[0]))[0][0]

        if gt_mode == "any":
            is_ok = int(pred in gt_lab)
            total += 1
            correct += is_ok
            for cid in gt_lab:
                per_cls_tot[cid] += 1
                if pred == cid:
                    per_cls_cor[cid] += 1
        else:
            total += 1
            per_cls_tot[int(gt_lab)] += 1
            is_ok = int(pred == int(gt_lab))
            correct += is_ok
            if is_ok:
                per_cls_cor[int(gt_lab)] += 1

    overall = float(correct) / float(total) if total else 0.0
    per_class = {cid: (float(per_cls_cor[cid]) / per_cls_tot[cid]) for cid in per_cls_tot if per_cls_tot[cid] > 0}
    
    return overall, per_class
