# augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(train: bool = True, size: int = 512, strength: str = "base"):
    if not train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(min_height=size, min_width=size, border_mode=0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.0,
            ),
            # (no MaskParams needed; masks are handled automatically in 1.1+)
        )

    if strength == "light":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=0, p=0.5),                   # bbox-safe
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05,      # bbox-safe
                               rotate_limit=10, border_mode=0, p=0.6),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            ToTensorV2(),
        ]
    elif strength == "strong":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),  # bbox-safe
                     rotate=(-20, 20), shear=(-8, 8), cval=0, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.12,      # bbox-safe
                               rotate_limit=20, border_mode=0, p=0.8),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=12, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.25),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.25),
            ToTensorV2(),
        ]
    else:  # "base"
        aug = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10,      # bbox-safe
                               rotate_limit=15, border_mode=0, p=0.7),
            A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.03), # bbox-safe
                     rotate=(-10, 10), shear=(-5, 5), cval=0, p=0.5),
            A.RandomBrightnessContrast(0.12, 0.12, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.2),
            ToTensorV2(),
        ]

    return A.Compose(
        aug,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )
