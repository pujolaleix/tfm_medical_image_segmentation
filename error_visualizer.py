# error_viz.py (replace visualize_errors_from_csv with this version)
import ast, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Dict
from matplotlib import font_manager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

_COL = {"TP": (66,133,244), "FP": (219,68,55), "FN": (244,180,0), "GT": (52,168,83)}

def _auto_styles(w, h, line_thickness=None, font_size=None, scale=1.0):
    base = min(w, h)
    if line_thickness is None:
        line_thickness = max(2, int(base * 0.006))
    if font_size is None:
        font_size = max(20, int(base * 0.040))
    return int(line_thickness * scale), int(font_size * scale)

def _load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    # explicit path
    if font_path:
        try: return ImageFont.truetype(font_path, size)
        except Exception: pass
    # PIL-bundled names
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(name, size)
        except Exception: pass
    # matplotlib fallback (usually works even in minimal envs)
    try:
        from matplotlib import font_manager
        path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    # common OS locations
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",                 # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "C:/Windows/Fonts/arial.ttf",                                     # Windows
    ):
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        except Exception:
            pass
        
    return ImageFont.load_default()

def _draw_box(draw, xyxy, color, width=4):
    x1,y1,x2,y2 = map(float, xyxy)
    try:
        draw.rectangle([x1,y1,x2,y2], outline=color, width=width)
    except TypeError:
        for i in range(width):
            draw.rectangle([x1-i,y1-i,x2+i,y2+i], outline=color)

def _draw_label(draw, xyxy, text, color, font, pad=4):
    x1,y1,_,_ = map(float, xyxy)
    tx, ty = x1 + 2, y1 + 2
    try:
        _, _, tw, th = draw.textbbox((tx,ty), text, font=font)
    except Exception:
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([tx-pad, ty-pad, tx+tw+pad, ty+th+pad], fill=(0,0,0))
    try:
        draw.text((tx,ty), text, fill=(255,255,255), font=font, stroke_width=1, stroke_fill=color)
    except TypeError:
        draw.text((tx,ty), text, fill=(255,255,255), font=font)

def _draw_label2(img, draw, xyxy, text, color, font, target_px: int, pad=4):
    x1, y1, _, _ = map(float, xyxy)

    # If this is a real FreeType font, draw normally (honors size)
    if font.__class__.__name__ == "FreeTypeFont":
        tx, ty = x1 + 2, y1 + 2
        try:
            _, _, tw, th = draw.textbbox((tx, ty), text, font=font)
        except Exception:
            tw, th = draw.textsize(text, font=font)
        draw.rectangle([tx-pad, ty-pad, tx+tw+pad, ty+th+pad], fill=(0,0,0))
        try:
            draw.text((tx, ty), text, fill=(255,255,255), font=font,
                      stroke_width=1, stroke_fill=color)
        except TypeError:
            draw.text((tx, ty), text, fill=(255,255,255), font=font)
        return

    # Bitmap fallback: render tiny then upscale to ~target_px height
    tmp = Image.new("RGBA", (1,1), (0,0,0,0))
    td  = ImageDraw.Draw(tmp)
    tw, th = td.textsize(text, font=font)
    txt = Image.new("RGBA", (tw+2*pad, th+2*pad), (0,0,0,255))
    td  = ImageDraw.Draw(txt)
    td.text((pad, pad), text, fill=(255,255,255,255), font=font)

    scale = max(1, int(round(target_px / max(th, 1))))
    big  = txt.resize((txt.width*scale, txt.height*scale), Image.NEAREST)
    img.paste(big, (int(x1)+2, int(y1)+2), big)

def _scale_box(xyxy, sx, sy):
    x1,y1,x2,y2 = map(float, xyxy)
    return [x1*sx, y1*sy, x2*sx, y2*sy]

def visualize_errors_from_csv(
    csv_path: str,
    images_root: str,
    out_dir: str,
    max_images_per_type: int = 10,
    font_path: str = None,
    line_thickness: Optional[int] = None,   # optional manual override
    font_size: Optional[int] = None,        # optional manual override
    style_scale: float = 1.0,     # multiply auto styles (e.g., 1.2)
    resize_max: Optional[int] = 1600 # longest side; None keeps original
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = {"TP":0, "FP":0, "FN":0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["type"]
            if t not in saved or saved[t] >= max_images_per_type:
                continue

            fn = row["file_name"]
            img_path = Path(images_root) / fn
            if not img_path.exists():
                cands = list(Path(images_root).rglob(Path(fn).name))
                if not cands: continue
                img_path = cands[0]

            img = Image.open(img_path).convert("RGB")
            ow, oh = img.size

            # optional downscale for visualization
            sx = sy = 1.0
            if resize_max is not None:
                scale = min(resize_max / max(ow, oh), 1.0)
                if scale < 1.0:
                    nw, nh = int(ow*scale), int(oh*scale)
                    img = img.resize((nw, nh), Image.BILINEAR)
                    sx = nw / ow; sy = nh / oh

            w, h = img.size
            lw, fs = _auto_styles(w, h, line_thickness, font_size, scale=style_scale)
            font = _load_font(font_path, fs)
            draw = ImageDraw.Draw(img)

            # draw GT
            if row["gt_bbox_xyxy"]:
                gt = _scale_box(ast.literal_eval(row["gt_bbox_xyxy"]), sx, sy)
                _draw_box(draw, gt, _COL["GT"], width=lw)

            # draw pred + labels
            if t in ("TP","FP") and row["pred_bbox_xyxy"]:
                pred = _scale_box(ast.literal_eval(row["pred_bbox_xyxy"]), sx, sy)
                _draw_box(draw, pred, _COL[t], width=lw)
                _draw_label2(img, draw, pred, f"{t} {row['category']}  s={row['score']}",
            _COL[t], font, target_px=fs)
            if t == "FN" and row["gt_bbox_xyxy"]:
                gt = _scale_box(ast.literal_eval(row["gt_bbox_xyxy"]), sx, sy)
                _draw_label(draw, gt, f"FN {row['category']}", _COL["FN"], font)

            out_name = f"{Path(fn).stem}_{t}.jpg"
            img.save(str(Path(out_dir)/out_name))
            saved[t] += 1

    return saved

