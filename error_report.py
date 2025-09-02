import json, csv
import numpy as np
from collections import defaultdict

def _xywh_to_xyxy(b):
    x,y,w,h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def _iou_xyxy(box, boxes):
    area  = max(box[2]-box[0], 0) * max(box[3]-box[1], 0)
    areas = np.clip(boxes[:,2]-boxes[:,0], 0, None) * np.clip(boxes[:,3]-boxes[:,1], 0, None)
    x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    return inter / (area + areas - inter + 1e-7)

def save_detection_error_report(gt_json, dt_json, out_csv, iou_thresh=0.50, max_dets_per_image=100):
    gt = json.load(open(gt_json, "r"))
    dt = json.load(open(dt_json, "r"))
    id2name = {c["id"]: c["name"] for c in gt.get("categories", [])}
    img_meta = {im["id"]: im for im in gt["images"]}

    gt_by_ic = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_ic[(int(a["image_id"]), int(a["category_id"]))].append(_xywh_to_xyxy(a["bbox"]))

    dt_by_ic = defaultdict(list)
    for d in dt:
        dt_by_ic[(int(d["image_id"]), int(d["category_id"]))].append(
            (float(d.get("score", 0.0)), _xywh_to_xyxy(d["bbox"]))
        )

    rows = []
    for key in set(gt_by_ic.keys()) | set(dt_by_ic.keys()):
        img_id, cat_id = key
        g_list = gt_by_ic.get(key, [])
        d_list = dt_by_ic.get(key, [])
        g = np.stack(g_list, axis=0) if g_list else np.zeros((0,4), np.float32)

        d_list.sort(key=lambda t: t[0], reverse=True)
        if max_dets_per_image is not None:
            d_list = d_list[:max_dets_per_image]

        used_g = set()
        for score, dbox in d_list:
            if g.shape[0] == 0:
                rows.append(["FP", img_meta[img_id]["file_name"], img_id, id2name.get(cat_id, str(cat_id)),
                             f"{score:.4f}", list(map(float, dbox)), "", "0.0"])
                continue
            ious = _iou_xyxy(dbox, g)
            if used_g:
                mask = np.zeros_like(ious, dtype=bool); mask[list(used_g)] = True; ious[mask] = -1.0
            j = int(np.argmax(ious)); val = float(ious[j])
            if val >= iou_thresh and j not in used_g:
                used_g.add(j)
                rows.append(["TP", img_meta[img_id]["file_name"], img_id, id2name.get(cat_id, str(cat_id)),
                             f"{score:.4f}", list(map(float, dbox)), list(map(float, g[j])), f"{val:.4f}"])
            else:
                rows.append(["FP", img_meta[img_id]["file_name"], img_id, id2name.get(cat_id, str(cat_id)),
                             f"{score:.4f}", list(map(float, dbox)), "", f"{val:.4f}"])

        # leftover GTs are FNs
        for j in range(len(g_list)):
            if j not in used_g:
                rows.append(["FN", img_meta[img_id]["file_name"], img_id, id2name.get(cat_id, str(cat_id)),
                             "", "", list(map(float, g[j])), ""])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type","file_name","image_id","category","score",
                    "pred_bbox_xyxy","gt_bbox_xyxy","IoU"])
        w.writerows(rows)
    return out_csv
