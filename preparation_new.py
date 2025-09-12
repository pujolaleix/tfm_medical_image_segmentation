
import os, json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Mapping
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from balancing import injury_label_remap


# ---------------------------
# Utilities
# ---------------------------

def polygon_to_bbox(poly: List[List[float]]) -> List[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [float(x0), float(y0), float(x1), float(y1)]  # xyxy

def polygon_to_mask(poly: List[List[float]], height: int, width: int) -> np.ndarray:
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in poly], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def safe_points_to_abs(val, W: int, H: int):
    pts = val.get("points")
    if pts is None:
        return None
    try:
        return [[float(x) * W / 100.0, float(y) * H / 100.0] for x, y in pts]
    except Exception:
        try:
            return [[float(p["x"]) * W / 100.0, float(p["y"]) * H / 100.0] for p in pts]
        except Exception:
            return None
        
def _region_center_xy(region):
    x1, y1, x2, y2 = region["bbox_xyxy"]
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _choose_one_region(regs, img_w: Optional[int], img_h: Optional[int], mode: str = "largest_area"):
    """
    Select a single region from `regs` (list of dicts with keys including: bbox_xyxy, area_px, num_points).
    Tie-breakers: more polygon points, then closer to image center.
    Supported modes:
      - "largest_area": pick ROI with max area_px (default)
      - "closest_to_center": pick ROI whose center is closest to image center
      - "merge_bbox": return a synthetic region whose bbox is the min/max envelope of all regs
    """
    if not regs:
        return None

    # Fallback image size from first region if missing
    W = img_w or regs[0].get("orig_w", None)
    H = img_h or regs[0].get("orig_h", None)

    if mode == "merge_bbox":
        # Envelope bbox enclosing all regions
        xs1 = [r["bbox_xyxy"][0] for r in regs]
        ys1 = [r["bbox_xyxy"][1] for r in regs]
        xs2 = [r["bbox_xyxy"][2] for r in regs]
        ys2 = [r["bbox_xyxy"][3] for r in regs]
        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
        area = max(0.0, (x2 - x1) * (y2 - y1))
        # Use the image-level label from the first region
        img_label = regs[0].get("image_label", regs[0].get("label", "unknown"))
        # Rectangle polygon (clockwise)
        rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return {
            "region_index": -1,
            "image_label": img_label,
            "region_label": regs[0].get("region_label", img_label),
            "points": rect_pts,
            "bbox_xyxy": [x1, y1, x2, y2],
            "orig_w": int(W) if W else None,
            "orig_h": int(H) if H else None,
            "num_points": 4,
            "area_px": float(area),
        }

    if mode == "closest_to_center" and (W is not None and H is not None):
        cx_img, cy_img = 0.5*W, 0.5*H
        def key_fun(r):
            cx, cy = _region_center_xy(r)
            d2 = (cx - cx_img)**2 + (cy - cy_img)**2
            # smaller distance preferred
            # tie-breakers: NEG num_points (more points), NEG area (larger)
            return (d2, -int(r.get("num_points", 0)), -float(r.get("area_px", 0.0)))
        return sorted(regs, key=key_fun)[0]

    # default: largest area with tie-breakers
    if mode == "largest_area":
        def key_fun_area(r):
            # NEG for descending area, then NEG num_points, then distance to center if available
            area = float(r.get("area_px", 0.0))
            npnt = int(r.get("num_points", 0))
            if W is not None and H is not None:
                cx_img, cy_img = 0.5*W, 0.5*H
                cx, cy = _region_center_xy(r)
                d2 = (cx - cx_img)**2 + (cy - cy_img)**2
            else:
                d2 = 0.0
            return (-area, -npnt, d2)
        return sorted(regs, key=key_fun_area)[0]
    
    return print("ROI selection mode not valid.")
        

def compute_class_weights_from_structured(train_s, label2id, background_idx=0, normalize=True, eps=1e-6):
    """
    Build weights [num_classes] aligned with your model's class indices:
      index 0 = background (fixed 1.0),
      index i = label2id[name] (i >= 1).
    Uses inverse frequency, optionally normalized to mean=1.
    """
    num_classes = 1 + len(label2id)  # background + foreground classes
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in train_s:
        cid = label2id[s["case_label"]]  # foreground id (>=1)
        counts[cid] += 1

    weights = np.ones(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if i == background_idx:
            weights[i] = 1.0
        else:
            weights[i] = 1.0 / max(counts[i], eps)

    if normalize:
        weights = weights * (len(weights) / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------
# Structured data preparation
# ---------------------------

def prepare_structured_dataset_df(
    df_roi: pd.DataFrame,
    df_meta: pd.DataFrame,
    mapping_target_label: Optional[pd.DataFrame],
    local_img_dir: str,
    stats_txt_path: str,
    label_field: str = "injury_label",
    downsampling_flag: bool = False,
    balance_skew: float = 0.5,
    balance_max_ratio: float = None,
    keep_one_bbox: bool = True,
    selection_mode: str = 'largest_area',
) -> List[Dict[str, Any]]:
    """Builds a list of dicts ready for our Dataset."""
    df_roi = df_roi.copy()
    df_meta = df_meta.copy()

    # --- normalize ids (CSV image column -> "pXXXX_YY.jpg")
    df_roi['image_id'] = df_roi['image'].apply(
        lambda x: os.path.basename(str(x)).rsplit('-', 1)[-1].lower()
    )

    # --- Excel rename & normalize
    df_meta = df_meta.rename(columns={
        'NombreFichero': 'image_id',
        'Diagnosis': 'injury_label',
        'Clasification': 'outcome_label'
    })
    df_meta['image_id'] = df_meta['image_id'].apply(
        lambda x: Path(str(x)).name.lower()
    )

    # --- keep only files that exist in the image dir
    img_dir = Path(local_img_dir)
    existing_files = {p.name.lower() for p in img_dir.iterdir() if p.is_file()}
    df_roi = df_roi[df_roi['image_id'].str.lower().isin(existing_files)].reset_index(drop=True)

    # --- merge
    df = pd.merge(df_roi, df_meta, on="image_id", how="left", validate="many_to_one")

    # --- data engineering for data balaning
    if (mapping_target_label is not None) and (label_field=="injury_label"):
        df = injury_label_remap(df, mapping_target_label)
        

    # --- patient id
    if 'Patient' in df.columns:
        df['patient_id'] = df['Patient'].astype(str).str.lower()
    else:
        df['patient_id'] = df['image_id'].str.extract(re.compile(r'(p\d{4})', re.IGNORECASE), expand=False).str.lower()

    out: List[Dict[str, Any]] = []
    per_image_counts: Dict[str, Counter] = defaultdict(Counter)
    all_labels = set()
    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_path = str(img_dir / image_id)

        lbl = row.get("label", None)
        if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
            continue

        try:
            label_json = json.loads(lbl) if isinstance(lbl, str) else lbl
        except Exception:
            continue
        if not isinstance(label_json, list) or len(label_json) == 0:
            continue

        first = label_json[0]
        W = int(first.get("original_width", first.get("value", {}).get("original_width", 0)) or 0)
        H = int(first.get("original_height", first.get("value", {}).get("original_height", 0)) or 0)

        regs = []
        for ridx, r in enumerate(label_json):
            val = r.get("value", r)
            w = int(r.get("original_width", val.get("original_width", W)) or W or 0)
            h = int(r.get("original_height", val.get("original_height", H)) or H or 0)
            if w <= 0 or h <= 0:
                continue

            poly_abs = safe_points_to_abs(val, w, h)
            if not poly_abs or len(poly_abs) < 3:
                continue

            bbox = polygon_to_bbox(poly_abs)
            area = max(0.0, (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))

            regs.append({
                "region_index": int(ridx),
                "label": row.get(label_field, "unknown"),
                "points": poly_abs,
                "bbox_xyxy": bbox,
                "orig_w": int(w),
                "orig_h": int(h),
                "num_points": int(len(poly_abs)),
                "area_px": float(area),
            })

        if not regs:
            continue

        if keep_one_bbox:
            chosen = _choose_one_region(regs, W, H, mode=selection_mode)
            if chosen is None:
                continue
            regs = [chosen]

        cls_counts = Counter([r["label"] for r in regs])
        per_image_counts[image_id].update(cls_counts)
        all_labels.update(cls_counts.keys())
        out.append({
            "image_id": image_id,
            "image_path": image_path,
            "width": int(W) if W else None,
            "height": int(H) if H else None,
            "patient_id": row.get("patient_id"),
            "case_label": row.get(label_field, "1"),
            "regions": regs
        })

    if stats_txt_path is not None:
        stats_txt_path = Path(stats_txt_path)
        stats_txt_path.parent.mkdir(parents=True, exist_ok=True)

        # consistent column order: sorted labels
        label_list = sorted(all_labels)
        with stats_txt_path.open("w", encoding="utf-8") as f:
            # header
            f.write("image_id\ttotal_boxes\t" + "\t".join(label_list) + "\n")
            # rows
            for image_id in sorted(per_image_counts.keys()):
                cnts = per_image_counts[image_id]
                total = sum(cnts.values())
                cols = [str(cnts.get(lbl, 0)) for lbl in label_list]
                multi = "True" if len([c for c in cnts.values() if c > 0]) > 1 else "False"
                f.write(f"{image_id}\t{total}\t" + "\t".join(cols) + f"\t{multi}\n")

        print(f"[prepare_structured_dataset_df] Wrote per-image bbox counts to: {stats_txt_path}")


    return out


def build_label_mapping(structured_data: List[Dict[str, Any]]):
    classes = sorted({r.get("label", "lesion")
                      for d in structured_data
                      for r in d.get("regions", [])})
    if not classes:
        classes = ["lesion"]
    label2id = {c: i + 1 for i, c in enumerate(classes)}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


# ---------------------------
# Dataset
# ---------------------------
class OralLesionDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], label2id: Dict[str, int], transform=None, strict_labels: bool = True):
        self.data = data
        self.transform = transform
        self.strict_labels = strict_labels

        # Build a normalized lookup for labels:  " Actinic   Cheilitis " -> "actinic cheilitis"
        def _norm(s: str) -> str:
            return " ".join(str(s).strip().lower().split())

        # label2id may already be 1..K; keep ids but normalize keys
        self.label2id_raw = label2id
        self.label2id = { _norm(k): v for k, v in label2id.items() }

        # assign a stable numeric id for each sample if not already there
        for i, d in enumerate(self.data):
            if "image_num_id" not in d:
                d["image_num_id"] = i

    def __len__(self):
        return len(self.data)

    def _lookup_label(self, lab: str) -> int:
        key = " ".join(str(lab).strip().lower().split())
        if key in self.label2id:
            return int(self.label2id[key])
        if self.strict_labels:
            # Be loud: this is the root cause of “all 1s”
            raise KeyError(f"Unknown label '{lab}' (normalized='{key}'). "
                           f"Provide it in label2id or fix the source data.")
        # Optional non-strict fallback: try exact (raw) then default to class 1 (not recommended)
        return int(self.label2id.get(key, 1))

    def __getitem__(self, idx: int):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        W, H = img.size

        boxes, labels, masks = [], [], []
        for reg in item.get("regions", []):
            # Prefer region label; optionally fall back to case/sample label if needed
            lab = reg.get("region_label", None)
            if lab is None:
                lab = reg.get("case_label", item.get("case_label", None))
            if lab is None:
                raise KeyError(f"No label found for sample idx={idx} image={item.get('image_path')}")

            cid = self._lookup_label(lab)

            bbox = reg.get("bbox_xyxy", None)
            if bbox is None:
                # derive from polygon
                poly = [[float(x), float(y)] for x, y in reg["points"]]
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                bbox = [min(xs), min(ys), max(xs), max(ys)]

            x0, y0, x1, y1 = bbox
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(W - 1.0, x1), min(H - 1.0, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
            labels.append(cid)

            # mask from polygon (clip to image)
            poly = reg.get("points", None)
            if poly is not None:
                poly = [[float(x), float(y)] for x, y in poly]
                poly = [[min(max(x, 0.0), W - 1.0), min(max(y, 0.0), H - 1.0)] for x, y in poly]
                m = polygon_to_mask(poly, H, W)
            else:
                # If no polygon, make a bbox mask
                m = np.zeros((H, W), dtype=np.uint8)
                m[int(y0):int(y1)+1, int(x0):int(x1)+1] = 1
            masks.append(m)

        if len(boxes) == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)
            masks_arr  = np.zeros((0, H, W), dtype=np.uint8)
        else:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)
            masks_arr  = np.stack(masks, axis=0).astype(np.uint8)

        # Keep original size for eval rescaling
        orig_H, orig_W = H, W

        image = np.array(img)

        # --- data augmentation / preprocessing
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes_arr,
                labels=labels_arr,
                masks=list(masks_arr),
            )
            image = transformed["image"]
            boxes_arr  = np.array(transformed["bboxes"], dtype=np.float32)
            labels_arr = np.array(transformed["labels"], dtype=np.int64)

            if len(boxes_arr) == 0:
                boxes_arr = np.zeros((0, 4), dtype=np.float32)
                labels_arr = np.zeros((0,), dtype=np.int64)
                masks_arr  = np.zeros((0, orig_H, orig_W), dtype=np.uint8)  # shape won't be used
            else:
                masks_arr = np.stack(transformed["masks"], axis=0).astype(np.uint8)

        # convert to tensors for PyTorch
        if isinstance(image, np.ndarray):
            image_t = torch.from_numpy(image).permute(2, 0, 1)
            if image_t.dtype == torch.uint8:
                image_t = image_t.float() / 255.0
            else:
                image_t = image_t.float()
        elif torch.is_tensor(image):
            image_t = image
            if image_t.dtype == torch.uint8:
                image_t = image_t.float() / 255.0
        else:
            raise TypeError(f"Unexpected image type: {type(image)}")

        # sizes for eval rescaling
        proc_H, proc_W = int(image_t.shape[-2]), int(image_t.shape[-1])

        target = {
            "boxes":  torch.as_tensor(boxes_arr, dtype=torch.float32),
            "labels": torch.as_tensor(labels_arr, dtype=torch.int64),
            "masks":  torch.as_tensor(masks_arr, dtype=torch.uint8),

            # numeric id for Mask R-CNN
            "image_id": torch.tensor(item["image_num_id"], dtype=torch.int64),

            # string key for evaluation (keep original)
            "image_key": item.get("image_id", item["image_path"]),

            # optional metadata
            "width": orig_W,
            "height": orig_H,
            "orig_size": torch.tensor([orig_H, orig_W], dtype=torch.int64),
            "proc_size": torch.tensor([proc_H, proc_W], dtype=torch.int64),
            "patient_id": item.get("patient_id"),
        }
        return image_t, target


# ---------------------------
# Split wrapper
# ---------------------------

def patient_level_split(structured_data: List[Dict[str, Any]],
                        ratios: Tuple[float, float, float],
                        label2id: Dict[str, int],
                        out_dir: str,
                        seed: int = 42):
    
    assert abs(sum(ratios) - 1.0) < 1e-6
    by_patient: Dict[str, List[Dict[str, Any]]] = {}

    for s in structured_data:
        pid = s.get("patient_id", "unknown")
        by_patient.setdefault(pid, []).append(s)

    patients = list(by_patient.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * ratios[0]))
    n_val   = int(round(n * ratios[1]))
    n_test  = n - n_train - n_val
    train_pat = set(patients[:n_train])
    val_pat   = set(patients[n_train:n_train+n_val])
    test_pat  = set(patients[n_train+n_val:])

    train_samples = [s for pid in train_pat for s in by_patient[pid]]
    val_samples   = [s for pid in val_pat   for s in by_patient[pid]]
    test_samples  = [s for pid in test_pat  for s in by_patient[pid]]

    #write
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "structured.json").open("w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump({
            "train_patients": sorted(list(train_pat)),
            "val_patients": sorted(list(val_pat)),
            "test_patients": sorted(list(test_pat)),
            "label2id": label2id,
        }, f, indent=2)

    create_hist(train_samples, val_samples, test_samples, out_dir / 'plots')

    return train_samples, val_samples, test_samples, (train_pat, val_pat, test_pat)


# collate
def detection_collate(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)



#create histogram from data splits.
def create_hist(train_split: List[Dict[str, Any]],
                val_split: List[Dict[str, Any]],
                test_split: List[Dict[str, Any]],
                out_dir: Union[str, Path],
                cmap_range:  tuple = (0.5, 1.0) 
                ):
    
    # Convert splits to DataFrames (dropping regions)
    def to_df(split, split_name):
        records = []
        for s in split:
            rec = {k: v for k, v in s.items() if k != "regions"}
            records.append(rec)
        df = pd.DataFrame(records)
        df["split"] = split_name
        return df

    df_train = to_df(train_split, "Train")
    df_val   = to_df(val_split,   "Validation")
    df_test  = to_df(test_split,  "Test")

    # Combine
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Counts table
    counts = df_all.groupby(["case_label", "split"]).size().unstack(fill_value=0)

    cmap = plt.cm.Blues
    colors = cmap(np.linspace(cmap_range[0], cmap_range[1], 3))

    # Plot horizontal stacked bars
    ax = counts.plot(kind="barh", stacked=True, figsize=(15, 15), color=colors)
    ax.set_ylabel("Case Label")
    ax.set_xlabel("Count")
    ax.set_title("Case label distribution by split")
    plt.legend(title="Split")  # Train, Validation, Test appear here
    plt.tight_layout()

    # Ensure output directory exists
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save files
    hist_path = out_path / "case_label_distribution.png"
    csv_path  = out_path / "case_label_distribution.csv"
    plt.savefig(hist_path, dpi=220)
    plt.close()

    counts.to_csv(csv_path, sep=",")

    print(f"Saved histogram to {hist_path}")
    print(f"Saved counts to {csv_path}")




# write metrics into txt file
def write_metrics_txt(metrics: Mapping[str, Any], path: str) -> str:
    """Pretty-print a nested metrics dict to a human-friendly .txt file."""
    def _lines(d: Mapping[str, Any], indent: str = ""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield f"{indent}{k}:"
                for line in _lines(v, indent + "  "):
                    yield line
            else:
                yield f"{indent}{k}: {v}"

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("\n".join(_lines(dict(metrics))) + "\n")
        
    return str(p)

def write_args_to_txt(args: dict, filepath: str):
    with open(filepath, "w") as f:
        for key, value in args.items():
            f.write(f"{key} = {value}\n")
