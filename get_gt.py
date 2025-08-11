import torch 
import numpy as np
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes


def get_gt(end_points):
    gt_labels = end_points['sem_cls_label']
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)
    box_label_mask = end_points['box_label_mask'] 
    gt_all_bbox = end_points['all_bboxes']
    all_bbox_label_mask = end_points['all_bbox_label_mask'] 
    auxi_box = end_points['auxi_box']
    
    gt_bbox_new = [gt_bbox[b, box_label_mask[b].bool()] for b in range(gt_labels.shape[0])]
    gt_bbox_new = [DepthInstance3DBoxes(box,box_dim=6, with_yaw=False, origin=(.5, .5, .5)) for box in gt_bbox_new]
    gt_labels_new = [gt_labels[b, box_label_mask[b].bool()] for b in range(gt_labels.shape[0])]
    gt_labels_new = [torch.zeros_like(tensor) for tensor in gt_labels_new]
    # img_meta = [{'box_type_3d': DepthInstance3DBoxes} for _ in range(len(gt_labels_new))]
    
    # Propagate negative sample information to img_meta so that
    # downstream loss computation (e.g. rejection loss) can know
    # which samples are negatives.  If `is_negative` is not provided
    # we default to ``False``.
    is_negative = end_points.get('is_negative', [False] * gt_labels.shape[0])
    # ensure list/array form for indexing
    if isinstance(is_negative, torch.Tensor):
        is_negative = is_negative.tolist()
    img_meta = [
        {'box_type_3d': DepthInstance3DBoxes, 'is_negative': bool(is_negative[b])}
        for b in range(len(gt_labels_new))
    ]
    
    gt_all_bbox_new = [gt_all_bbox[b, all_bbox_label_mask[b].bool()] for b in range(gt_labels.shape[0])]
    gt_all_bbox_new = [DepthInstance3DBoxes(box,box_dim=6, with_yaw=False, origin=(.5, .5, .5)) for box in gt_all_bbox_new]
    
    auxi_bbox = [DepthInstance3DBoxes(auxi_box[b],box_dim=6, with_yaw=False, origin=(.5, .5, .5)) if auxi_box[b].max()>0
                 else DepthInstance3DBoxes([],box_dim=6, with_yaw=False, origin=(.5, .5, .5)) for b in range(gt_labels.shape[0])]

    return gt_bbox_new, gt_labels_new, gt_all_bbox_new, auxi_bbox, img_meta