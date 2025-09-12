from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import io

from contextlib import contextmanager, redirect_stderr, redirect_stdout


@contextmanager
def suppress_stdout_stderr():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


# ----------------------- COCOeval runner (helper) -----------------------

def run_cocoeval(
    gt_json_path: str,
    dt_json_path: str,
    coco_iouType: str = "bbox"
):
    """
    Loads GT/DT jsons, runs COCOeval.evaluate/accumulate/summarize and returns the COCOeval object.
    Use iouType='bbox' for bbox-only detections, 'segm' when your DT also contains 'segmentation'.
    """

    with suppress_stdout_stderr():

        coco_gt = COCO(str(gt_json_path))
        dt_bbox = json.load(open(dt_json_path, "r"))
        coco_dt = coco_gt.loadRes(dt_bbox)

        E = COCOeval(coco_gt, coco_dt, iouType=coco_iouType)

        E.evaluate()
        E.accumulate()
        E.summarize()

    return E


# ----------------------- IoU / Dice from COCOeval -----------------------


@torch.no_grad()
def image_level_label_accuracy(
    gt_json_path: str,
    dt_json_path: str,
    agg: str = "max",          # "max" or "sum" over detection scores per class
    score_thresh: float = 0.0, # ignore detections below this score
):
    """
    Image-level classification accuracy ignoring localization.
    GT label = majority class among GT annotations in that image.
    Pred label = class with max (or sum) detection score across that image's detections.
    """
    coco_gt = COCO(str(gt_json_path))
    # GT: image_id -> majority class id
    gt_labels_per_img = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        gt_labels_per_img[ann["image_id"]].append(int(ann["category_id"]))
    gt_majority = {}
    for img_id, labs in gt_labels_per_img.items():
        if labs:
            gt_majority[img_id] = Counter(labs).most_common(1)[0][0]

    # DT: group by image, aggregate per class
    dt = json.load(open(dt_json_path, "r"))
    dets_by_img = defaultdict(list)
    for d in dt:
        if d.get("score", 0.0) >= score_thresh:
            dets_by_img[int(d["image_id"])].append((int(d["category_id"]), float(d["score"])))

    total = correct = 0
    per_cls_tot = Counter()
    per_cls_cor = Counter()

    # evaluate only images that have a GT majority label
    for img_id, gt_cls in gt_majority.items():
        scores = dets_by_img.get(img_id, [])
        if not scores:
            pred_cls = -1  # "no prediction"
        else:
            if agg == "sum":
                acc = defaultdict(float)
                for c, s in scores: 
                    acc[c] += s
                pred_cls = max(acc.items(), key=lambda kv: kv[1])[0]
            else:  # "max"
                best = {}
                for c, s in scores:
                    if (c not in best) or (s > best[c]): best[c] = s
                pred_cls = max(best.items(), key=lambda kv: kv[1])[0]

        total += 1
        per_cls_tot[gt_cls] += 1
        if pred_cls == gt_cls:
            correct += 1
            per_cls_cor[gt_cls] += 1

    img_cls_acc = correct / max(1, total)
    per_class_acc = {int(k): (per_cls_cor[k] / per_cls_tot[k]) for k in per_cls_tot if per_cls_tot[k] > 0}

    return img_cls_acc, per_class_acc, total




@torch.no_grad()
def iou_and_dice_from_cocoeval(
    eval_obj,
    thr: float = 0.50,
    area_label: str = "all",
) -> Tuple[float, Dict[int, float], float, Dict[int, float]]:
    """
    One-pass computation of IoU and Dice (overall + per-class) from a finished COCOeval object.
    - Uses the IoU threshold closest to `thr`
    - Uses the area slice named by `area_label` (default 'all') and the largest maxDet bucket
    Returns: (overall_iou, per_class_iou, overall_dice, per_class_dice)
    """
    iouThrs = np.asarray(eval_obj.params.iouThrs, dtype=float)
    t_idx = int(np.argmin(np.abs(iouThrs - float(thr))))

    area_lbls = list(getattr(eval_obj.params, "areaRngLbl", ["all"]))
    a_idx = area_lbls.index(area_label) if area_label in area_lbls else 0
    a_rng = eval_obj.params.areaRng[a_idx]
    max_det = eval_obj.params.maxDets[-1]

    per_cat_ious: Dict[int, list] = {}
    all_ious = []

    for ev in (eval_obj.evalImgs or []):
        if not ev:
            continue
        if list(ev.get("aRng", [])) != list(a_rng):
            continue
        if int(ev.get("maxDet", max_det)) != int(max_det):
            continue

        imgId = int(ev["image_id"])
        catId = int(ev.get("category_id", -1))
        if catId < 0:
            continue

        arr = eval_obj.ious.get((imgId, catId))
        if arr is None:
            arr = eval_obj.ious.get((catId, imgId))
        if arr is None:
            continue

        ious_mat = np.asarray(arr)
        if ious_mat.size == 0:
            continue

        dtMatches = np.asarray(ev["dtMatches"])[t_idx]           # [D]
        dtIgnore  = np.asarray(ev["dtIgnore"])[t_idx].astype(bool)  # [D]
        gtIds     = [int(g) for g in ev.get("gtIds", [])]
        if not gtIds:
            continue

        gt_index = {gid: j for j, gid in enumerate(gtIds)}
        vals = per_cat_ious.setdefault(catId, [])

        for di, gid in enumerate(map(int, dtMatches)):
            if gid == 0:
                continue
            if di < len(dtIgnore) and bool(dtIgnore[di]):
                continue
            j = gt_index.get(gid)
            if j is None:
                continue
            iou = float(ious_mat[di, j])
            if iou <= 0.0 or np.isnan(iou):
                continue
            vals.append(iou)
            all_ious.append(iou)

    per_class_iou = {cid: float(np.mean(v)) for cid, v in per_cat_ious.items() if v}
    overall_iou = float(np.mean(all_ious)) if all_ious else 0.0

    to_dice = lambda x: (2.0 * x) / (1.0 + x)
    per_class_dice = {cid: float(np.mean([to_dice(x) for x in v])) for cid, v in per_cat_ious.items() if v}
    overall_dice = float(np.mean([to_dice(x) for x in all_ious])) if all_ious else 0.0

    return overall_iou, per_class_iou, overall_dice, per_class_dice


def mean_iou_from_cocoeval(eval_obj: COCOeval, thr: float):

    # nearest IoU threshold index
    t_idx = int(np.argmin(np.abs(eval_obj.params.iouThrs - float(thr))))
    per_cat_vals = {}  # {catId: [iou_value, ...]}

    for ev in eval_obj.evalImgs:
        if ev is None:
            continue
        # match COCO AP settings: default area range + largest maxDet
        if ev["aRng"] != eval_obj.paramss.areaRng[0] or ev["maxDet"] != eval_obj.params.maxDets[-1]:
            continue

        imgId = ev["image_id"]
        catId = ev["category_id"]

        # IoUs for this (img, cat): may be list or ndarray
        arr = eval_obj.ious.get((imgId, catId))
        if arr is None:
            continue
        ious_mat = np.asarray(arr)
        if ious_mat.size == 0:
            continue

        # threshold-specific matches/ignores; can be lists or arrays
        dtMatches = ev["dtMatches"][t_idx]   # [D] matched gtId (or 0)
        dtIgnore  = ev["dtIgnore"][t_idx]    # [D] flags
        gtIds     = ev["gtIds"]              # list of gt ids (ints)

        # build a fast map: gtId -> column index in ious_mat
        gt_index = {int(g): j for j, g in enumerate(gtIds)}

        vals = per_cat_vals.setdefault(catId, [])
        # iterate detections in this image/category
        for di, gid in enumerate(dtMatches):
            gid = int(gid)  # could be float 0.0
            if gid == 0:
                continue
            if bool(dtIgnore[di]):  # works for 0/1, bool, or np.bool_
                continue
            j = gt_index.get(gid, None)
            if j is None:
                continue
            vals.append(float(ious_mat[di, j]))

    per_cat_mean = {cid: float(np.mean(v)) for cid, v in per_cat_vals.items() if len(v)}
    all_vals = [x for v in per_cat_vals.values() for x in v]
    overall = float(np.mean(all_vals)) if all_vals else 0.0
    
    return overall, per_cat_mean



def dice_from_cocoeval(e: COCOeval, thr: float = 0.50):
    """
    Compute mean Dice from a finished COCOeval object by converting matched IoUs:
      Dice = 2*IoU / (1+IoU)
    Returns (overall_mean_dice, per_class_mean_dice_dict)
    """
    t_idx = int(np.argmin(np.abs(e.params.iouThrs - float(thr))))
    per_cat_vals = {}

    for ev in e.evalImgs:
        if not ev:
            continue
        # Match the headline AP slice: area="all", maxDet = largest
        if ev["aRng"] != e.params.areaRng[0] or ev["maxDet"] != e.params.maxDets[-1]:
            continue

        imgId, catId = ev["image_id"], ev["category_id"]
        arr = e.ious.get((imgId, catId))
        if arr is None:
            continue
        ious_mat = np.asarray(arr)
        if ious_mat.size == 0:
            continue

        dtMatches = ev["dtMatches"][t_idx]
        dtIgnore  = ev["dtIgnore"][t_idx]
        gtIds     = list(ev["gtIds"])
        gt_index  = {int(g): j for j, g in enumerate(gtIds)}
        vals = per_cat_vals.setdefault(int(catId), [])

        for di, gid in enumerate(dtMatches):
            gid = int(gid)
            if gid == 0 or bool(dtIgnore[di]):
                continue
            j = gt_index.get(gid)
            if j is None:
                continue
            iou = float(ious_mat[di, j])
            dice = (2.0 * iou) / (1.0 + iou) if iou > 0.0 else 0.0
            vals.append(dice)

    per_class = {cid: float(np.mean(v)) for cid, v in per_cat_vals.items() if v}
    all_vals = [x for v in per_cat_vals.values() for x in v]
    overall = float(np.mean(all_vals)) if all_vals else 0.0
    return overall, per_class

