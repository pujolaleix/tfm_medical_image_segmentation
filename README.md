# Deep Learning for Oral Lesion Detection and Segmentation: A Flexible, Clinically Grounded Pipeline

A compact pipeline for **object detection + instance segmentation** (Mask R-CNN) on medical images (oral lesions).  
It covers **dataset prep, augmentation, training, evaluation, visualization**, and **balancing**. The primary entrypoint for running experiments is the **`main2.ipynb`** notebook.
This is the code delivered to support my thesis memory for master in Computer Vision in UAB.
---

## 📦 What’s inside

```
augmentations.py         # Albumentations pipelines (light/base/strong)
balancing.py             # Class balancing & label remapping helpers
coco_io.py               # COCO JSON I/O for GT/DT, helpers
evaluation_coco.py       # Metrics using pycocotools (IoU, Dice, AP)
evaluation_no_coco.py    # Metrics without pycocotools
inference_visuals.py     # Visual GT vs DT overlays (boxes + masks)
losses2.py               # Dice / Focal / BCE + custom RoI heads
metrics.py               # High-level metric runner (bbox & mask)
model2_final.py          # Training/val loop, scheduler, splits
models.py                # Model builders (Mask R-CNN + custom heads)
preparation_new.py       # Data loading/structuring, class weights
main2.ipynb              # Notebook entrypoint (main is called here)
```

---

### 1) Environment
```bash
torch>=2.1
torchvision>=0.16
albumentations>=1.3
pycocotools>=2.0
pandas>=2.0
numpy>=1.24
Pillow>=9.5
tqdm>=4.65
matplotlib>=3.7
```

### 2) Data layout - [not included in the Repo]
- Images folder, e.g. `data/images/`
- ROI annotations table (CSV/JSON) and metadata table (CSV/Excel)
- (Optional) label mapping for superclass remaps

> `preparation_new.py` expects to merge ROI annotations with metadata, normalize filenames, derive patient IDs, and build a per-image list of regions with **polygon**, **bbox**, and **label** info.

### 3) Run experiments (recommended)
Open **`main2.ipynb`** and run cells in order. The notebook calls into the same functions used by the scripts, so you can tweak aug strength (`light`, `base`, `strong`), loss mix (Dice/BCE/Focal), patient-level split ratios, sampler settings, etc.

### 4) Script usage (optional)
Most notebook steps are also available as scripts. Example training call:
```bash
python model2_final.py   --images_dir ./data/images   --df_roi ./data/roi.csv   --df_meta ./data/meta.csv   --output_dir ./outputs   --epochs 25   --augmentation_type base   --pretrained true
```
This will: prepare the dataset, split at **patient level**, build Mask R-CNN with custom heads, train with AMP + grad clipping, and save metrics/figures under `./outputs/`.

### 5) Evaluation & visualization
- **Metrics**: Use `metrics.py` or functions from `evaluation_*` to compute mAP/mAR/IoU/Dice and image-level accuracy from COCO JSONs.
- **Visuals**: `inference_visuals.py` renders GT (green) vs detections (red) with crisp mask overlays and per-instance labels. Example:
```python
from inference_visuals import visualize_gt_dt_samples
visualize_gt_dt_samples(
    gt_json_path="path/to/instances_val.json",
    dt_bbox_json_path="path/to/detections_boxes.json",
    dt_segm_json_path="path/to/detections_masks.json",
    images_dir="path/to/images",
    out_dir="viz_out",
    score_thresh=0.3,
    num_images=12,
)
```

---

## 🧠 Key design notes
- **COCO-first**: Utilities to export GT/DT to COCO format for standard tooling.
- **Robust evaluation**: COCO-based and no-COCO fallbacks for reproducibility.
- **Custom heads & losses**: Mask R-CNN with dropout box head, weighted CE, Dice/Focal mixing.
- **Balancing**: Downsampling to control class skew and optional superclass remaps.
- **Augmentation**: Albumentations implementation as data augmentation.
- **Patient-level splits**: Avoids leakage between train/val/test.

---

