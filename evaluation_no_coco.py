import json
import heapq
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
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
    max_dets: int = 100,
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


def bbox_average_precision_recall_from_json(
    gt_json_path: str,
    dt_json_path: str,
    iou_thr: float = 0.50,
    score_thresh: float = 0.0,
    max_dets: int = 100,
    return_curves: bool = False,
) -> Tuple[float, Dict[int, float], float, Dict[int, float], Optional[Dict[int, Dict[str, np.ndarray]]]]:
    """
    Average Precision/Recall for bounding boxes at a single IoU threshold.
    - AP is area under the precision-recall curve (continuous interpolation).
    - Recall is max recall (TP / #GT) after sweeping all detections.

    Returns:
      mAP, per_class_AP, mAR, per_class_recall, curves (optional)
      where curves[cid] = {"precision": np.ndarray, "recall": np.ndarray, "scores": np.ndarray}
    """
    _, _, _, gts, dts, cat_ids, img_ids = _index_coco_jsons(gt_json_path, dt_json_path)

    per_class_AP: Dict[int, float] = {}
    per_class_AR: Dict[int, float] = {}
    curves: Dict[int, Dict[str, np.ndarray]] = {}

    for cat_id in cat_ids:
        # #GT instances for this class
        npos = sum(len(gts.get((img_id, cat_id), [])) for img_id in img_ids)
        if npos == 0:
            continue

        # Collect detections across images (respect per-image max_dets and score_thresh)
        dets = []
        for img_id in img_ids:
            dt_list = [d for d in dts.get((img_id, cat_id), []) if float(d.get("score", 0.0)) >= score_thresh]
            if max_dets is not None and len(dt_list) > max_dets:
                dt_list = dt_list[:max_dets]  # they are already sorted by score DESC in _index_coco_jsons
            for d in dt_list:
                dets.append((img_id, float(d.get("score", 0.0)), _bbox_xywh_to_xyxy(d["bbox"])))

        if not dets:
            per_class_AP[cat_id] = 0.0
            per_class_AR[cat_id] = 0.0
            if return_curves:
                curves[cat_id] = {"precision": np.array([0.0]), "recall": np.array([0.0]), "scores": np.array([])}
            continue

        # Sort globally by score DESC
        dets.sort(key=lambda t: -t[1])
        scores = np.array([s for _, s, _ in dets], dtype=np.float32)

        # Prepare GT used flags per image
        gt_xyxy = {
            img_id: np.array([_bbox_xywh_to_xyxy(ann["bbox"]) for ann in gts.get((img_id, cat_id), [])], dtype=np.float32)
            for img_id in img_ids
        }
        used = { (img_id, i): False
                 for img_id in img_ids
                 for i in range(len(gts.get((img_id, cat_id), []))) }

        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        # Greedy matching per detection (global score order)
        for j, (img_id, _, dbox) in enumerate(dets):
            g = gt_xyxy.get(img_id)
            if g is None or len(g) == 0:
                fp[j] = 1.0
                continue
            ious = np.array([_iou_xyxy(dbox, gk) for gk in g], dtype=np.float32)
            best = int(np.argmax(ious))
            best_iou = float(ious[best])
            key = (img_id, best)
            if best_iou >= iou_thr and not used[key]:
                tp[j] = 1.0
                used[key] = True
            else:
                fp[j] = 1.0

        # PR curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(npos)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # AP via precision envelope (COCO/VOC2010 style)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        per_class_AP[cat_id] = ap
        per_class_AR[cat_id] = float(rec[-1] if rec.size else 0.0)
        if return_curves:
            curves[cat_id] = {"precision": prec, "recall": rec, "scores": scores}

    valid = [cid for cid in cat_ids if cid in per_class_AP]
    mAP = float(np.mean([per_class_AP[cid] for cid in valid])) if valid else 0.0
    mAR = float(np.mean([per_class_AR[cid] for cid in valid])) if valid else 0.0
    return mAP, per_class_AP, mAR, per_class_AR, (curves if return_curves else None)


def mask_average_precision_recall_from_json(
    gt_json_path: str,
    dt_json_path: str,
    iou_thr: float = 0.50,
    score_thresh: float = 0.0,
    max_dets: int = 100,
    return_curves: bool = False,
) -> Tuple[float, Dict[int, float], float, Dict[int, float], Optional[Dict[int, Dict[str, np.ndarray]]]]:
    """
    Average Precision/Recall for instance masks at a single IoU threshold.
    Uses greedy 1-1 matching within each image. Requires pycocotools.
    """
    if not _HAVE_MASK:
        raise RuntimeError("pycocotools is required for mask AP/AR.")

    gt, _, img_hw, gts, dts, cat_ids, img_ids = _index_coco_jsons(gt_json_path, dt_json_path)

    per_class_AP: Dict[int, float] = {}
    per_class_AR: Dict[int, float] = {}
    curves: Dict[int, Dict[str, np.ndarray]] = {}

    for cat_id in cat_ids:
        # #GT instances for this class (with segmentation)
        npos = 0
        for img_id in img_ids:
            npos += sum(1 for g in gts.get((img_id, cat_id), []) if "segmentation" in g)
        if npos == 0:
            continue

        all_scores, all_tp, all_fp = [], [], []

        for img_id in img_ids:
            h, w = img_hw[img_id]
            gt_list = [g for g in gts.get((img_id, cat_id), []) if "segmentation" in g]
            dt_list = [d for d in dts.get((img_id, cat_id), []) if "segmentation" in d and float(d.get("score", 0.0)) >= score_thresh]
            if max_dets is not None and len(dt_list) > max_dets:
                dt_list = dt_list[:max_dets]  # already sorted by score DESC

            if not dt_list:
                continue

            # IoU matrix for this image/class (rows=dets in score order, cols=gts)
            iou_mat = _mask_iou_matrix(dt_list, gt_list, h, w)  # shape [D, G]
            used = np.zeros(iou_mat.shape[1], dtype=bool)       # matched GT flags

            for d_idx, d in enumerate(dt_list):
                score = float(d.get("score", 0.0))
                all_scores.append(score)
                if iou_mat.shape[1] == 0:
                    all_fp.append(1.0); all_tp.append(0.0)
                    continue
                # best unmatched GT for this detection
                ious = iou_mat[d_idx]
                best = int(np.argmax(ious))
                best_iou = float(ious[best])
                if (best_iou >= iou_thr) and (not used[best]):
                    used[best] = True
                    all_tp.append(1.0); all_fp.append(0.0)
                else:
                    all_tp.append(0.0); all_fp.append(1.0)

        if not all_scores:
            per_class_AP[cat_id] = 0.0
            per_class_AR[cat_id] = 0.0
            if return_curves:
                curves[cat_id] = {"precision": np.array([0.0]), "recall": np.array([0.0]), "scores": np.array([])}
            continue

        order = np.argsort(-np.asarray(all_scores, dtype=np.float32))
        tp = np.asarray(all_tp, dtype=np.float32)[order]
        fp = np.asarray(all_fp, dtype=np.float32)[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(npos)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        per_class_AP[cat_id] = ap
        per_class_AR[cat_id] = float(rec[-1] if rec.size else 0.0)
        if return_curves:
            curves[cat_id] = {"precision": prec, "recall": rec, "scores": np.asarray(all_scores)[order]}

    valid = [cid for cid in cat_ids if cid in per_class_AP]
    mAP = float(np.mean([per_class_AP[cid] for cid in valid])) if valid else 0.0
    mAR = float(np.mean([per_class_AR[cid] for cid in valid])) if valid else 0.0
    return mAP, per_class_AP, mAR, per_class_AR, (curves if return_curves else None)


# -------------------- MASK IoU & Dice (JSON-only, if segm present) --------------------

def mask_iou_dice_from_json(
    gt_json_path: str,
    dt_json_path: str,
    iou_thr: float = 0.50,
    score_thresh: float = 0.0,
    max_dets: int = 100,
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
    score_thresh: float = 0.0,     # filter detections by score
    gt_mode: str = "majority",     # "majority" | "any"
    count_blank_as_wrong: bool = True,
    max_dets: Optional[int] = 1,   # cap #detections per image (after score filter). None = no cap
) -> Tuple[float, Dict[int, float]]:
    """
    Build a single predicted class per image from detections and compare to GT image label.

    - agg='max': pick the class of the single highest-scoring detection in the image.
    - agg='sum': sum scores per class, then argmax (useful if you expect many duplicates).
    - gt_mode='majority': GT label is the most frequent category in that image.
      gt_mode='any': GT set = all categories in image; prediction is correct if it's in the set.
    - max_dets: per-image cap *after* score filtering; we keep the top-`max_dets` detections by score.
    - Special case: if an image has no detections, it is counted as correct IFF its GT is "healthy".
      The "healthy" class id is discovered from gt["categories"] by name (case-insensitive).
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(dt_json_path, "r", encoding="utf-8") as f:
        dt = json.load(f)

    # Find "healthy" class id(s) from GT categories (case-insensitive exact name match)
    healthy_ids: Set[int] = set()
    for cat in gt.get("categories", []):
        name = str(cat.get("name", "")).strip().lower()
        if name == "healthy":
            healthy_ids.add(int(cat["id"]))

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
            # tie-breaker: higher count first, then smaller class id
            top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            gt_targets[img_id] = int(top)

    # Collect detections per image (score-filtered)
    dts_by_img = defaultdict(list)
    for d in dt:
        s = float(d.get("score", 0.0))
        if s >= score_thresh:
            dts_by_img[int(d["image_id"])].append((int(d["category_id"]), s))

    # Optional: cap per image by top-k scores
    if max_dets is not None:
        for img_id, lst in dts_by_img.items():
            if len(lst) > max_dets:
                dts_by_img[img_id] = heapq.nlargest(max_dets, lst, key=lambda kv: kv[1])

    total = 0
    correct = 0
    per_cls_tot = Counter()
    per_cls_cor = Counter()

    for img_id, gt_lab in gt_targets.items():
        scores = dts_by_img.get(img_id, [])

        # handle blank predictions
        if not scores:
            total += 1

            if gt_mode == "majority":
                per_cls_tot[int(gt_lab)] += 1
                is_healthy_gt = int(gt_lab) in healthy_ids
            else:  # gt_mode == "any" -> gt_lab is a set
                is_healthy_gt = bool(healthy_ids.intersection(gt_lab))

            if is_healthy_gt:
                # Blank is considered correct for healthy GT (overrides count_blank_as_wrong)
                correct += 1
                if gt_mode == "majority":
                    per_cls_cor[int(gt_lab)] += 1
            else:
                # Fall back to original blank-handling
                if not count_blank_as_wrong:
                    correct += 1
                    if gt_mode == "majority":
                        per_cls_cor[int(gt_lab)] += 1
            continue

        # choose predicted class
        if agg == "sum":
            acc = defaultdict(float)
            for cid, s in scores:
                acc[cid] += s
            pred = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:  # "max"
            pred = max(scores, key=lambda kv: (kv[1], -kv[0]))[0]  # highest score, tie-breaker: smaller class id

        # score
        if gt_mode == "any":
            # gt_lab is a set
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