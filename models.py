import torch

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from losses2 import CustomRoIHeads, TwoMLPHeadWithDropout
from preparation_new import compute_class_weights_from_structured


# base mask-rcnn
def create_maskrcnn(num_classes: int, pretrained: bool = True):
    # num_classes includes background
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model



# freeze few blocks
def freeze_stem_layer1(m):
    for p in m.backbone.body.conv1.parameters(): p.requires_grad = False
    for p in m.backbone.body.bn1.parameters():   p.requires_grad = False
    for p in m.backbone.body.layer1.parameters(): p.requires_grad = False


# custom roi mask-rcnn
def build_model(
    num_classes: int,
    class_weights: torch.Tensor = None,
    dice_alpha: float = 0.5,
    pretrained: bool = True,
    use_bce: bool = True,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    dropout_alpha: float = 0.2,
):
    """
    Build Mask R-CNN with multi-class heads:
      - box classifier/regressor sized to `num_classes` (bg included)
      - mask predictor sized to `num_classes` (bg included)
    Then wraps with CustomRoIHeads preserving the numeric hyperparams.
    """

    base = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)

    # ---- Replace BOX head with correct number of classes ----
    in_features = base.roi_heads.box_predictor.cls_score.in_features
    base.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ---- Replace MASK head with correct number of classes ----
    in_features_mask = base.roi_heads.mask_predictor.conv5_mask.in_channels
    base.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    # (Optional) quick sanity checks
    assert base.roi_heads.box_predictor.cls_score.out_features == num_classes, \
        "Box classifier output dim mismatch."
    assert base.roi_heads.mask_predictor.mask_fcn_logits.out_channels == num_classes, \
        "Mask predictor output dim mismatch."

    # ---- Extract numeric args your CustomRoIHeads expects ----
    pm = getattr(base.roi_heads, "proposal_matcher", None)
    sampler = getattr(base.roi_heads, "fg_bg_sampler", None)
    if pm is None or sampler is None:
        raise RuntimeError("Could not find proposal_matcher / fg_bg_sampler on base.roi_heads.")

    fg_iou_thresh       = float(pm.high_threshold)
    bg_iou_thresh       = float(pm.low_threshold)
    batch_size_per_img  = int(sampler.batch_size_per_image)
    positive_fraction   = float(sampler.positive_fraction)

    # bbox_reg_weights may be on roi_heads or box_coder
    if getattr(base.roi_heads, "bbox_reg_weights", None) is not None:
        brw = base.roi_heads.bbox_reg_weights
    else:
        brw = base.roi_heads.box_coder.weights
    bbox_reg_weights = tuple(brw.tolist() if isinstance(brw, torch.Tensor) else brw)

    score_thresh       = float(base.roi_heads.score_thresh)
    nms_thresh         = float(base.roi_heads.nms_thresh)
    detections_per_img = int(base.roi_heads.detections_per_img)

    # (Optional) validate class_weights shape if provided
    if class_weights is not None:
        if class_weights.numel() not in (num_classes, num_classes - 1):
            raise ValueError(
                f"class_weights must have length {num_classes} (incl. bg) "
                f"or {num_classes - 1} (foreground only); got {class_weights.numel()}."
            )

    base.roi_heads = CustomRoIHeads(
        box_roi_pool        = base.roi_heads.box_roi_pool,
        box_head            = TwoMLPHeadWithDropout(base.roi_heads.box_head, p=dropout_alpha),
        box_predictor       = base.roi_heads.box_predictor,
        fg_iou_thresh       = fg_iou_thresh,
        bg_iou_thresh       = bg_iou_thresh,
        batch_size_per_image= batch_size_per_img,
        positive_fraction   = positive_fraction,
        bbox_reg_weights    = bbox_reg_weights,
        score_thresh        = score_thresh,
        nms_thresh          = nms_thresh,
        detections_per_img  = detections_per_img,
        mask_roi_pool       = base.roi_heads.mask_roi_pool,
        mask_head           = base.roi_heads.mask_head,
        mask_predictor      = base.roi_heads.mask_predictor,
        class_weights       = class_weights,
        dice_alpha          = dice_alpha,
        use_bce             = use_bce,
        use_focal           = use_focal,
        focal_alpha         = focal_alpha,
        focal_gamma         = focal_gamma,
        label_smoothing     = label_smoothing,
    )

    return base



# Function to select model configuration
def model_selector(train_s, label2id, device, model_type, pretrained, num_classes, max_class_weights, dice_alpha, use_bce, use_focal, focal_alpha, focal_gamma, label_smoothing, dropout_alpha):

    if model_type=='base':
        print('Base Mask-RCNN model selected...')
        model = create_maskrcnn(num_classes=num_classes, pretrained=pretrained).to(device)

    elif model_type=='custom_roi':
        print('Mask-RCNN with CustomROI and weighted loss model selected...')
        class_weights = compute_class_weights_from_structured(
                            train_s, label2id=label2id, background_idx=0, normalize=True
                        ).to(device)
        class_weights = torch.clamp(class_weights, max=max_class_weights) #[2-0, 5.0]

        model = build_model(
            num_classes=num_classes,
            class_weights=class_weights,
            dice_alpha=dice_alpha,          # mix between (BCE/Focal) and Dice
            pretrained=pretrained,
            use_bce=use_bce,            # enable BCE
            use_focal=use_focal,         # also add Focal (optional)
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            dropout_alpha=dropout_alpha,
        ).to(device)

    else: print(f"{model_type} configuration not defined.")

    return model