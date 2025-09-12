# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads


# ----------------------------
# Dice loss (binary, per-ROI)
# ----------------------------
class DiceLossFromLogits(nn.Module):
    """Computes Dice loss from raw logits."""
    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, H, W) or (N, 1, H, W)
        targets: same shape, binary {0,1}
        """
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(1)
        dice = (2 * inter + self.eps) / (probs.sum(1) + targets.sum(1) + self.eps)
        return 1.0 - dice.mean()


# ----------------------------
# Binary Focal Loss
# ----------------------------
class FocalLoss(nn.Module):
    """Binary focal loss for masks."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: logits (N, H, W)
        targets: binary {0,1}, same shape
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        foc = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return foc.mean()
        elif self.reduction == "sum":
            return foc.sum()
        return foc



# ----------------------------
# Dropout ROI Head
# ----------------------------
class TwoMLPHeadWithDropout(nn.Module):
    def __init__(self, base_head: nn.Module, p: float = 0.2):
        super().__init__()
        # reuse original layers so weights/shapes stay correct
        self.fc6 = base_head.fc6
        self.fc7 = base_head.fc7
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # IMPORTANT: flatten ROI pooled features from (N, C, 7, 7) -> (N, C*7*7)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        return x


# ----------------------------
# Custom ROIHeads (torchvision 0.15.0a0 signature)
# ----------------------------
class CustomROIHeads(RoIHeads):
    """
    RoIHeads with:
      - Weighted CE for classifier (uses class_weights)
      - Flexible mask loss: BCE and/or Focal combined with Dice via dice_alpha
    """
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_thresh: float,         # numeric, NOT a Matcher
        bg_iou_thresh: float,
        batch_size_per_image: int,    # numeric, NOT a sampler
        positive_fraction: float,
        bbox_reg_weights,             # tuple/list/Tensor of 4 floats
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        *,
        class_weights: torch.Tensor = None,  # shape [num_classes], index 0=background
        dice_alpha: float = 0.5,             # 0..1: weight on (BCE/Focal) vs Dice
        use_bce: bool = True,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        # Forward EXACT numeric args expected by your torchvision build
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )

        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.dice_alpha = float(dice_alpha)
        self.use_bce = bool(use_bce)
        self.use_focal = bool(use_focal)
        self.dice_fn = DiceLossFromLogits()
        self.focal_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing = float(label_smoothing)

    # --- Weighted CE for classification + SmoothL1 for boxes ---
    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        if self.class_weights is not None:
            cls_loss = F.cross_entropy(class_logits, labels, weight=self.class_weights, label_smoothing=self.label_smoothing)
        else:
            cls_loss = F.cross_entropy(class_logits, labels, label_smoothing=self.label_smoothing)

        sampled_pos_inds = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds]

        N, _ = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds, labels_pos],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            reduction="sum",
        ) / labels.numel()

        return cls_loss, box_loss

    # --- BCE/Focal + Dice for masks (per positive ROI) ---
    def maskrcnn_loss(self, mask_logits, proposals, gt_masks, gt_labels):
        pos_inds = torch.where(gt_labels > 0)[0]
        if pos_inds.numel() == 0 or mask_logits.numel() == 0:
            return mask_logits.sum() * 0

        labels_pos = gt_labels[pos_inds]
        mask_logits = mask_logits[pos_inds, labels_pos]       # (Npos, M, M)
        gt_masks = gt_masks[pos_inds].to(dtype=mask_logits.dtype)

        # Build components
        ce_focal = 0.0
        count=0
        if self.use_bce:
            ce_focal = ce_focal + F.binary_cross_entropy_with_logits(mask_logits, gt_masks)
            count += 1
        if self.use_focal:
            ce_focal = ce_focal + self.focal_fn(mask_logits, gt_masks)
            count += 1
        if count > 0:
            ce_focal = ce_focal / count

        dice = self.dice_fn(mask_logits.unsqueeze(1), gt_masks.unsqueeze(1))

        # If neither BCE nor Focal is enabled, fallback to Dice-only
        if not (self.use_bce or self.use_focal):
            return dice

        return self.dice_alpha * ce_focal + (1.0 - self.dice_alpha) * dice

CustomRoIHeads = CustomROIHeads