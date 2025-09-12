from preparation_new import write_metrics_txt
from coco_io import coco_stats_dict
from evaluation_no_coco import *
from evaluation_coco import *

from pathlib import Path


def run_metrics(
        gt_json_path,
        dt_bbox_path,
        dt_segm_path,
        out_dir,
        id2label: Dict[int, str],
        epoch: int = None,
        iou_thr: float = 0.50,
        dice_thr: float = 0.50,
        score_thr: float = 0.00,
        img_acc_score_thr: float = 0.05,
        agg_acc_im: str = 'max',
        segm_flag: bool = False,
        loss_metric: str = 'val',
        mode_type: str = 'train',
        ):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #bbox
    miou, percls_iou, mdice, percls_dice = bbox_iou_dice_from_json(gt_json_path, dt_bbox_path, 
                                                                   iou_thr=iou_thr, score_thresh=score_thr)
    per_class_iou = {id2label[cid]: float(percls_iou.get(cid, -2.0)) for cid in sorted(id2label.keys())}
    per_class_dice = {id2label[cid]: float(percls_dice.get(cid, -2.0)) for cid in sorted(id2label.keys())}

    mAP, ap_per_cls, mAR, ar_per_cls, curves = bbox_average_precision_recall_from_json(gt_json_path, dt_bbox_path, 
                                                                                       iou_thr=iou_thr, score_thresh=score_thr, return_curves=True)
    per_class_map = {id2label[cid]: float(ap_per_cls.get(cid, -2.0)) for cid in sorted(id2label.keys())}
    per_class_mar = {id2label[cid]: float(ar_per_cls.get(cid, -2.0)) for cid in sorted(id2label.keys())}

    img_acc, percls_imgacc = image_level_label_accuracy_from_json(gt_json_path, dt_bbox_path, agg=agg_acc_im, 
                                                                  score_thresh=img_acc_score_thr, max_dets=1, gt_mode="majority")
    per_class_imgacc = {id2label[cid]: float(percls_imgacc.get(cid, -2.0)) for cid in sorted(id2label.keys())}


    e_bbox = run_cocoeval(gt_json_path, dt_bbox_path, coco_iouType='bbox')
    coco_miou, per_cls_iou, coco_mdice, per_cls_dice = iou_and_dice_from_cocoeval(e_bbox, thr=iou_thr)
    coco_per_class_iou = {id2label[cid]: float(per_cls_iou.get(cid, -2.0)) for cid in sorted(id2label.keys())}
    coco_per_class_dice = {id2label[cid]: float(per_cls_dice.get(cid, -2.0)) for cid in sorted(id2label.keys())}
    
    coco_img_acc, coco_percls_imgacc, _ = image_level_label_accuracy(gt_json_path, dt_bbox_path, agg=agg_acc_im, score_thresh=img_acc_score_thr)
    coco_per_class_acc_img = {id2label[cid]: float(coco_percls_imgacc.get(cid, -2.0)) for cid in sorted(id2label.keys())}

    
    bbox_metrics = {
        f"nococo_bbox_IoU@{iou_thr:.2f}_overall": float(miou),
        f"nococo_bbox_Dice@{dice_thr:.2f}_overall": float(mdice),
        f"nococo_bbox_IoU@{iou_thr:.2f}_per_class": per_class_iou,
        f"nococo_bbox_Dice@{dice_thr:.2f}_per_class": per_class_dice,
        f"nococo_bbox_mAP": float(mAP),
        f"nococo_bbox_mAR": float(mAR),
        f"nococo_bbox_mAP_per_class": per_class_map,
        f"nococo_bbox_mAR_per_class": per_class_mar,

        # Image-level accuracy
        f"img_level_acc@{img_acc_score_thr:.2f}": float(img_acc),
        f"img_accuracy_per_class@{img_acc_score_thr:.2f}": per_class_imgacc,

        # COCOeval bbox
        f"coco_bbox_IoU@{iou_thr:.2f}_overall": float(coco_miou),
        f"coco_bbox_Dice@{dice_thr:.2f}_overall": float(coco_mdice),
        f"coco_bbox_IoU@{iou_thr:.2f}_per_class": coco_per_class_iou,
        f"coco_bbox_Dice@{dice_thr:.2f}_per_class": coco_per_class_dice,
        **coco_stats_dict(e_bbox),
        f"coco_image_level_accuracy@{img_acc_score_thr:.2f}": coco_img_acc,
        f"coco_img_accuracy_per_class@{img_acc_score_thr:.2f}" : coco_per_class_acc_img
    }

    if mode_type=='train':
        write_metrics_txt(bbox_metrics, str(out_dir/f"metrics_ckpt/val_metrics_epoch{epoch+1:02d}.txt"))
        print(f"Metrics saved at {out_dir}/metrics_ckpt")
    elif mode_type=='test':
        write_metrics_txt(bbox_metrics, str(out_dir/"metrics_bbox.txt"))

    #segm - masks
    if segm_flag or (mode_type=='test'):
        
        s_miou, s_percls_iou, s_mdice, s_percls_dice = mask_iou_dice_from_json(gt_json_path, dt_segm_path, 
                                                                   iou_thr=iou_thr, score_thresh=score_thr)
        s_per_class_iou = {id2label[cid]: float(s_percls_iou.get(cid, -2.0)) for cid in sorted(id2label.keys())}
        s_per_class_dice = {id2label[cid]: float(s_percls_dice.get(cid, -2.0)) for cid in sorted(id2label.keys())}

        s_mAP, s_ap_per_cls, s_mAR, s_ar_per_cls, s_curves = mask_average_precision_recall_from_json(gt_json_path, dt_segm_path, 
                                                                                       iou_thr=iou_thr, score_thresh=score_thr, return_curves=True)
        s_per_class_map = {id2label[cid]: float(s_ap_per_cls.get(cid, -2.0)) for cid in sorted(id2label.keys())}
        s_per_class_mar = {id2label[cid]: float(s_ar_per_cls.get(cid, -2.0)) for cid in sorted(id2label.keys())}

        e_segm = run_cocoeval(gt_json_path, dt_segm_path, coco_iouType='segm')

        s_coco_miou, s_per_cls_iou, s_coco_mdice, s_per_cls_dice = iou_and_dice_from_cocoeval(e_segm, thr=iou_thr)
        s_coco_per_class_iou = {id2label[cid]: float(s_per_cls_iou.get(cid, -2.0)) for cid in sorted(id2label.keys())}
        s_coco_per_class_dice = {id2label[cid]: float(s_per_cls_dice.get(cid, -2.0)) for cid in sorted(id2label.keys())}

        
        segm_metrics = {
            f"nococo_segm_IoU@{iou_thr:.2f}_overall": float(s_miou),
            f"nococo_segm_Dice@{dice_thr:.2f}_overall": float(s_mdice),
            f"nococo_segm_IoU@{iou_thr:.2f}_per_class": s_per_class_iou,
            f"nococo_segm_Dice@{dice_thr:.2f}_per_class": s_per_class_dice,
            f"nococo_segm_mAP": float(s_mAP),
            f"nococo_segm_mAR": float(s_mAR),
            f"nococo_segm_mAP_per_class": s_per_class_map,
            f"nococo_segm_mAR_per_class": s_per_class_mar,

            # COCOeval segm
            f"coco_segm_IoU@{iou_thr:.2f}_overall": float(s_coco_miou),
            f"coco_segm_Dice@{dice_thr:.2f}_overall": float(s_coco_mdice),
            f"coco_segm_IoU@{iou_thr:.2f}_per_class": s_coco_per_class_iou,
            f"coco_segm_Dice@{dice_thr:.2f}_per_class": s_coco_per_class_dice,
            **coco_stats_dict(e_segm),
        }

        if mode_type=='train':
            write_metrics_txt(segm_metrics, str(out_dir/f"metrics_ckpt/val_metrics_segm_epoch{epoch+1:02d}.txt"))
            print(f"Metrics saved at {out_dir}/metrics_ckpt")
        elif mode_type=='test':
            write_metrics_txt(segm_metrics, str(out_dir/"metrics_mask.txt"))
            print(f"Metrics saved at {out_dir}")
                  
    

    if (loss_metric=='dice'):
        metric_out = mdice
    elif (loss_metric=='iou'):
        metric_out = miou
    elif (loss_metric=='ap'):
        metric_out = mAP
    else:
        metric_out = None
        return metric_out

    print(f"loss metric: {loss_metric} - {metric_out:.2f}")
    
    return metric_out