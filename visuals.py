import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from preparation_new import OralLesionDataset
import random
import os

def visualize_and_save_augmentations(dataset, save_dir="augmented_samples", num_samples=10):
    """
    Apply dataset transforms and save augmented samples as PNGs.

    Args:
        dataset: an instance of OralLesionDataset (with transform defined)
        save_dir (str): folder where images will be saved
        num_samples (int): how many random samples to save
    """
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        image, target = dataset[idx]

        # convert CHW torch image back to HWC numpy [0,255]
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        # draw boxes
        for box in target["boxes"]:
            x0, y0, x1, y1 = box.int().tolist()
            cv2.rectangle(img_np, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # draw masks (semi-transparent)
        if "masks" in target and len(target["masks"]) > 0:
            mask = target["masks"][0].cpu().numpy().astype("uint8") * 255
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            img_np = cv2.addWeighted(img_np, 0.7, colored_mask, 0.3, 0)

        # save to disk
        out_path = os.path.join(save_dir, f"augmented_{target['image_id'].item()}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        #print(f"Saved {out_path}")
