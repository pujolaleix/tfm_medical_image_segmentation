from preparation_new import write_metrics_txt
from coco_io import coco_stats_dict
from evaluation_no_coco import *
from evaluation_coco import *


def run_metrics(
        gt_json_path,
        dt_bbox_path,
        dt_segm_path,
        out_dir,
        id2label: Dict[int, str],
        epoch: int,
        iou_thr: float = 0.50,
        dice_thr: float = 0.50,
        score_thr: float = 0.00,
        mask_thr: float = 0.00,
        agg_acc_im: str = 'max',
        segm_flag: bool = False,
        loss_metric: str = 'val',
        ):

    #bbox
    miou, percls_iou, mdice, percls_dice = bbox_iou_dice_from_json(gt_json_path, dt_bbox_path, 
                                                                   iou_thr=iou_thr, score_thresh=score_thr)
    per_class_iou = {id2label[cid]: float(percls_iou.get(cid, 0.0)) for cid in sorted(id2label.keys())}
    per_class_dice = {id2label[cid]: float(percls_dice.get(cid, 0.0)) for cid in sorted(id2label.keys())}

    img_acc, percls_imgacc = image_level_label_accuracy_from_json(gt_json_path, dt_bbox_path, agg=agg_acc_im, 
                                                                  score_thresh=score_thr, gt_mode="majority")
    per_class_imgacc = {id2label[cid]: float(percls_imgacc.get(cid, 0.0)) for cid in sorted(id2label.keys())}


    e_bbox = run_cocoeval(gt_json_path, dt_bbox_path, coco_iouType='bbox')
    coco_miou, per_cls_iou, coco_mdice, per_cls_dice = iou_and_dice_from_cocoeval(e_bbox, thr=iou_thr)
    coco_per_class_iou = {id2label[cid]: float(per_cls_iou.get(cid, 0.0)) for cid in sorted(id2label.keys())}
    coco_per_class_dice = {id2label[cid]: float(per_cls_dice.get(cid, 0.0)) for cid in sorted(id2label.keys())}
    
    coco_img_acc, coco_percls_imgacc, _ = image_level_label_accuracy(gt_json_path, dt_bbox_path, agg=agg_acc_im, score_thresh=score_thr)
    coco_per_class_acc_img = {id2label[cid]: float(coco_percls_imgacc.get(cid, 0.0)) for cid in sorted(id2label.keys())}

    
    bbox_metrics = {
        f"nococo_bbox_IoU@{iou_thr:.2f}_overall": float(miou),
        f"nococo_bbox_Dice@{dice_thr:.2f}_overall": float(mdice),
        f"nococo_bbox_IoU@{iou_thr:.2f}_per_class": per_class_iou,
        f"nococo_bbox_Dice@{dice_thr:.2f}_per_class": per_class_dice,

        # Image-level accuracy
        "img_level_acc": float(img_acc),
        "img_accuracy_per_class": per_class_imgacc,

        # COCOeval bbox
        f"coco_bbox_IoU@{iou_thr:.2f}_overall": float(coco_miou),
        f"coco_bbox_Dice@{dice_thr:.2f}_overall": float(coco_mdice),
        f"coco_bbox_IoU@{iou_thr:.2f}_per_class": coco_per_class_iou,
        f"coco_bbox_Dice@{dice_thr:.2f}_per_class": coco_per_class_dice,
        **coco_stats_dict(e_bbox),
        "coco_image_level_accuracy": coco_img_acc,
        "coco_img_accuracy_per_class" : coco_per_class_acc_img
    }

    write_metrics_txt(bbox_metrics, str(out_dir/f"metrics_ckpt/val_metrics_epoch{epoch+1:02d}.txt"))

    #segm - masks
    if segm_flag:
        
        s_miou, s_percls_iou, s_mdice, s_percls_dice = mask_iou_dice_from_json(gt_json_path, dt_segm_path, 
                                                                   iou_thr=iou_thr, score_thresh=mask_thr)
        s_per_class_iou = {id2label[cid]: float(s_percls_iou.get(cid, 0.0)) for cid in sorted(id2label.keys())}
        s_per_class_dice = {id2label[cid]: float(s_percls_dice.get(cid, 0.0)) for cid in sorted(id2label.keys())}

        e_segm = run_cocoeval(gt_json_path, dt_segm_path, coco_iouType='segm')

        s_coco_miou, s_per_cls_iou, s_coco_mdice, s_per_cls_dice = iou_and_dice_from_cocoeval(e_segm, thr=iou_thr)
        s_coco_per_class_iou = {id2label[cid]: float(s_per_cls_iou.get(cid, 0.0)) for cid in sorted(id2label.keys())}
        s_coco_per_class_dice = {id2label[cid]: float(s_per_cls_dice.get(cid, 0.0)) for cid in sorted(id2label.keys())}
        dice_test, _ = dice_from_cocoeval(e_segm, thr=iou_thr)

        
        segm_metrics = {
            f"nococo_bbox_IoU@{iou_thr:.2f}_overall": float(s_miou),
            f"nococo_bbox_Dice@{dice_thr:.2f}_overall": float(s_mdice),
            f"nococo_bbox_IoU@{iou_thr:.2f}_per_class": s_per_class_iou,
            f"nococo_bbox_Dice@{dice_thr:.2f}_per_class": s_per_class_dice,
            "DICE TEST": dice_test,

            # COCOeval segm
            f"coco_bbox_IoU@{iou_thr:.2f}_overall": float(s_coco_miou),
            f"coco_bbox_Dice@{dice_thr:.2f}_overall": float(s_coco_mdice),
            f"coco_bbox_IoU@{iou_thr:.2f}_per_class": s_coco_per_class_iou,
            f"coco_bbox_Dice@{dice_thr:.2f}_per_class": s_coco_per_class_dice,
            **coco_stats_dict(e_segm),
        }

        write_metrics_txt(segm_metrics, str(out_dir/f"metrics_ckpt/val_metrics_segm_epoch{epoch+1:02d}.txt"))

    print(f"Metrics saved at {out_dir}/metrics_ckpt")

    if (loss_metric=='dice'):
        metric_out = mdice
    elif (loss_metric=='iou'):
        metric_out = miou
    elif (loss_metric=='ap'):
        metric_out = bbox_metrics['AP']
    else:
        metric_out = None
        return metric_out

    print(f"loss metric: {loss_metric} - {metric_out:.2f}")
    
    return metric_out