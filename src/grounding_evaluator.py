# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc
import numpy as np

def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

# BRIEF Evaluator
class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(
        self,
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1, 5, 10],
        prefixes=[],
        filter_non_gt_boxes=False,
        voxel_size=1.0,
        dim_is_radius=False,
        axis_perm=(0, 1, 2),
        axis_sign=(1, 1, 1),
        use_scene_offset=False,
        offset_keys=("scene_offset", "origin", "pc_min", "shift", "scene_shift"),
        gt_in_world=True,
        debug=False,
    ):
        """Initialize accumulators and configuration parameters."""
        """Initialize accumulators.

        Args:
            only_root (bool): detect only the root noun
            thresholds (list): IoU thresholds to check
            topks (list): k to evaluate top--k accuracy
            prefixes (list): names of layers to evaluate
            filter_non_gt_boxes (bool): whether to filter predictions
                that do not correspond to GT boxes
            voxel_size (float): Deprecated. Predictions are assumed to be
                in world coordinates; the argument is ignored but kept for
                backward compatibility.
        """
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes
        self.filter_non_gt_boxes = filter_non_gt_boxes
        self.voxel_size = voxel_size
        self.dim_is_radius = dim_is_radius
        self.axis_perm = axis_perm
        self.axis_sign = axis_sign
        self.use_scene_offset = use_scene_offset
        self.offset_keys = offset_keys
        self.gt_in_world = gt_in_world
        self.debug = debug
        self._debug_shown = False
        self.reset()

    def _get_scene_offset(self, end_points, bid, device):
        """Retrieve per-scene offset if provided in ``end_points``."""
        for key in self.offset_keys:
            if key in end_points:
                off = end_points[key]
                if isinstance(off, (list, tuple)):
                    off = torch.tensor(off, device=device, dtype=torch.float32)
                if isinstance(off, torch.Tensor):
                    if off.ndim == 2:
                        return off[bid].to(device)
                    if off.ndim == 1:
                        return off.to(device)
        return torch.zeros(3, device=device)

    def _restore_pred_boxes(self, boxes, end_points, bid):
        """Restore predicted boxes to world coordinates."""
        device = boxes.device
        center, dims = boxes[..., :3], boxes[..., 3:]

        perm = list(self.axis_perm)
        center = center[..., perm]
        dims = dims[..., perm]

        sign = torch.tensor(self.axis_sign, device=device, dtype=torch.float32)
        center = center * sign

        if self.dim_is_radius:
            dims = dims * 2

        if self.use_scene_offset:
            offset = self._get_scene_offset(end_points, bid, device)
            center = center + offset

        return torch.cat([center, dims], dim=-1)

    def _prep_gt_boxes(self, gt_boxes, end_points=None, bid=None):
        """Convert GT boxes to (cx,cy,cz,w,h,d)."""
        boxes = torch.cat([gt_boxes.gravity_center, gt_boxes.dims], dim=1)
        if not self.gt_in_world:
            boxes = self._restore_pred_boxes(boxes, end_points, bid)
        return boxes

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, k, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for k in self.topks
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})
        self.dets.update({'vd50': 0, 'vid50': 0})
        self.dets.update({'hard50': 0, 'easy50': 0})
        self.dets.update({'multi50': 0, 'unique50': 0})
        self.gts.update({'vd50': 1e-14, 'vid50': 1e-14})
        self.gts.update({'hard50': 1e-14, 'easy50': 1e-14})
        self.gts.update({'multi50': 1e-14, 'unique50': 1e-14})

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'position alignment',
            'bbf': 'semantic alignment'
        }
        for prefix in self.prefixes:
            for mode in ['bbf']:
                for t in self.thresholds:
                    print(
                        prefix, 'Acc%.2f:' % t,
                        ', '.join([
                            'Top-%d: %.5f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
        print('\nAnalysis')
        print('iou@0.25')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            print(field, self.dets[field] / self.gts[field])
        print('iou@0.50')
        for field in ['easy50', 'hard50', 'vd50', 'vid50', 'unique50', 'multi50']:
            print(field, self.dets[field] / self.gts[field])

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    # BRIEF Evaluation
    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # NOTE Two Evaluation Ways: position alignment, semantic alignment
        # self.evaluate_bbox_by_pos_align(end_points, prefix)
        # self.evaluate_bbox_by_sem_align(end_points, prefix)
        
        self.evaluate_bbox_by_3dcnn(end_points, prefix)

    def evaluate_bbox_by_3dcnn(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        max_k = max(self.topks)

        # Highest scoring box -> iou
        for bid in range(len(end_points['bbox_results'])):
            gt_bboxes = self._prep_gt_boxes(
                end_points['gt_bboxes_3d'][bid], end_points, bid
            )
            scores = end_points['bbox_results'][bid]['scores_3d'].reshape(-1)
            pred_boxes = torch.cat([
                end_points['bbox_results'][bid]['bboxes_3d'].gravity_center,
                end_points['bbox_results'][bid]['bboxes_3d'].dims,
            ], dim=-1).reshape(-1, 6)

            num_boxes = scores.shape[0]
            if num_boxes >= max_k:
                top_idx = torch.topk(scores, max_k).indices
                pbox = pred_boxes[top_idx]
            else:
                pad = torch.zeros(max_k - num_boxes, 6, device=pred_boxes.device)
                pbox = torch.cat([pred_boxes, pad], dim=0)

            pbox = self._restore_pred_boxes(pbox, end_points, bid)
            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes),
                box_cxcyczwhd_to_xyzxyz(pbox).to(gt_bboxes.device),
            )

            if self.debug and not self._debug_shown and bid == 0:
                gt_iou, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(gt_bboxes),
                    box_cxcyczwhd_to_xyzxyz(gt_bboxes),
                )
                pred_iou, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(pbox),
                    box_cxcyczwhd_to_xyzxyz(pbox),
                )
                print(
                    f"[DEBUG] IoU(gt,gt) mean={gt_iou.mean():.4f} max={gt_iou.max():.4f}"
                )
                print(
                    f"[DEBUG] IoU(pred,pred) mean={pred_iou.mean():.4f} max={pred_iou.max():.4f}"
                )
                print(
                    f"[DEBUG] GT center range {gt_bboxes[:, :3].min(0)[0]} to {gt_bboxes[:, :3].max(0)[0]}"
                )
                print(
                    f"[DEBUG] Pred center range {pbox[:, :3].min(0)[0]} to {pbox[:, :3].max(0)[0]}"
                )
                print(
                    f"[DEBUG] GT dims range {gt_bboxes[:, 3:].min(0)[0]} to {gt_bboxes[:, 3:].max(0)[0]}"
                )
                print(
                    f"[DEBUG] Pred dims range {pbox[:, 3:].min(0)[0]} to {pbox[:, 3:].max(0)[0]}"
                )
                diff = pbox[:, :3] - gt_bboxes[:, :3].mean(0)
                print(
                    f"[DEBUG] Pred-GT center diff mean {diff.mean(0)}"
                )
                self._debug_shown = True

            # step Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(dim=1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += thresholded.size(0)
                    if prefix == '3dcnn' and k == 1:
                        hit = found[0].item()
                        if t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += hit
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += hit
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += hit
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += hit
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += hit
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += hit
                        if t == self.thresholds[1]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd50'] += 1
                                self.dets['vd50'] += hit
                            else:
                                self.gts['vid50'] += 1
                                self.dets['vid50'] += hit
                            if end_points['is_hard'][bid]:
                                self.gts['hard50'] += 1
                                self.dets['hard50'] += hit
                            else:
                                self.gts['easy50'] += 1
                                self.dets['easy50'] += hit
                            if end_points['is_unique'][bid]:
                                self.gts['unique50'] += 1
                                self.dets['unique50'] += hit
                            else:
                                self.gts['multi50'] += 1
                                self.dets['multi50'] += hit


    
    # BRIEF position alignment
    def evaluate_bbox_by_pos_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q=256, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1) # ([B, 256, 6])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            top = scores.argsort(1, True)[:, :10]
            pbox = pred_bbox[bid, top.reshape(-1)]

            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]   # ([1, 10])

            # step Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)

    # BRIEF semantic alignment
    def evaluate_bbox_by_sem_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)

        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # step Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += len(thresholded)
                    if prefix == 'last_':
                        found = found[0].item()
                        if k == 1 and t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += found
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += found
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += found
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += found
                        if k == 1 and t == self.thresholds[1]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd50'] += 1
                                self.dets['vd50'] += found
                            else:
                                self.gts['vid50'] += 1
                                self.dets['vid50'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard50'] += 1
                                self.dets['hard50'] += found
                            else:
                                self.gts['easy50'] += 1
                                self.dets['easy50'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique50'] += 1
                                self.dets['unique50'] += found
                            else:
                                self.gts['multi50'] += 1
                                self.dets['multi50'] += found


    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_center = end_points['center_label'][:, :, 0:3]       
        gt_size = end_points['size_gts']                        
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)     # GT box cxcyczwhd
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]        # (B, 1, 6)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_bboxes
    
import torch.distributed as dist
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
class RejectionGroundingEvaluator:
    """Comprehensive evaluator for grounding and rejection."""

    def __init__(
        self,
        iou_thresh=0.5,
        rejection_thresh=0.5,
        voxel_size=1.0,
        dim_is_radius=False,
        axis_perm=(0, 1, 2),
        axis_sign=(1, 1, 1),
        use_scene_offset=False,
        offset_keys=("scene_offset", "origin", "pc_min", "shift", "scene_shift"),
        gt_in_world=True,
        debug=False,
    ):
        self.iou_thresh = iou_thresh
        self.rejection_thresh = rejection_thresh
        self.voxel_size = voxel_size
        self.dim_is_radius = dim_is_radius
        self.axis_perm = axis_perm
        self.axis_sign = axis_sign
        self.use_scene_offset = use_scene_offset
        self.offset_keys = offset_keys
        self.gt_in_world = gt_in_world
        self.debug = debug
        self._debug_shown = False
        self.reset()

    def reset(self):
        """重置所有计数器"""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.pos_count = 0
        self.neg_count = 0
    # ------------------------------------------------------------------
    # Helpers for coordinate restoration
    def _get_scene_offset(self, end_points, bid, device):
        for key in self.offset_keys:
            if key in end_points:
                off = end_points[key]
                if isinstance(off, (list, tuple)):
                    off = torch.tensor(off, device=device, dtype=torch.float32)
                if isinstance(off, torch.Tensor):
                    if off.ndim == 2:
                        return off[bid].to(device)
                    if off.ndim == 1:
                        return off.to(device)
        return torch.zeros(3, device=device)

    def _restore_pred_boxes(self, boxes, end_points, bid):
        device = boxes.device
        center, dims = boxes[:, :3], boxes[:, 3:]
        perm = list(self.axis_perm)
        center = center[:, perm]
        dims = dims[:, perm]
        sign = torch.tensor(self.axis_sign, device=device).view(1, 3)
        center = center * sign
        if self.dim_is_radius:
            dims = dims * 2
        if self.use_scene_offset:
            offset = self._get_scene_offset(end_points, bid, device).view(1, 3)
            center = center + offset
        return torch.cat([center, dims], dim=-1)

    def _prep_gt_boxes(self, gt_boxes, end_points=None, bid=None):
        boxes = torch.cat([gt_boxes.gravity_center, gt_boxes.dims], dim=1)
        if not self.gt_in_world:
            boxes = self._restore_pred_boxes(boxes, end_points, bid)
        return boxes

    def evaluate(self, end_points, prefix):
        """
        Evaluate on a batch.
        """
        gt_bboxes_list = end_points['gt_bboxes_3d']
        bbox_results_list = end_points['bbox_results']
        is_negative_list = end_points.get('is_negative',
                                      [False] * len(end_points['gt_bboxes_3d']))

        for i in range(len(is_negative_list)):
            is_negative = is_negative_list[i]

            if not is_negative:
                # --- Handle positive samples (localization task) ---
                self.pos_count += 1
                pred_results = bbox_results_list[i]

                if pred_results['scores_3d'].shape[0] == 0:
                    self.fn += 1
                    if i == 0 and dist.get_rank() == 0:
                        print(f"\n[DEBUG] Positive Sample | No prediction made. Counted as FN.")
                    continue

                best_score_idx = torch.argmax(pred_results['scores_3d'])
                best_score = pred_results['scores_3d'][best_score_idx]
                if i == 0 and dist.get_rank() == 0:
                    # Use .item() to convert the tensor to a Python float
                    print(f"\n[DEBUG] Positive Sample | Max Confidence: {best_score.item():.4f}")

                gt_bbox = self._prep_gt_boxes(gt_bboxes_list[i], end_points, i)
                pred_box = torch.cat([
                    pred_results['bboxes_3d'][best_score_idx:best_score_idx+1].gravity_center,
                    pred_results['bboxes_3d'][best_score_idx:best_score_idx+1].dims,
                ], dim=1)
                pred_box = self._restore_pred_boxes(pred_box, end_points, i)
                if not self.gt_in_world:
                    gt_bbox = self._restore_pred_boxes(gt_bbox, end_points, i)
                gt_box_ends = box_cxcyczwhd_to_xyzxyz(gt_bbox)
                pred_box_ends = box_cxcyczwhd_to_xyzxyz(pred_box)

                iou, _ = _iou3d_par(gt_box_ends.to(pred_box_ends.device), pred_box_ends)

                if torch.max(iou) >= self.iou_thresh:
                    self.tp += 1
                else:
                    self.fn += 1

            else:
                # --- Handle negative samples (rejection task) ---
                self.neg_count += 1
                pred_results = bbox_results_list[i]

                if pred_results['scores_3d'].shape[0] == 0:
                    self.tn += 1
                    if i == 0 and dist.get_rank() == 0:
                        print(f"\n[DEBUG] Negative Sample | No prediction made. Correctly Rejected (TN).")
                    continue

                max_confidence = torch.max(pred_results['scores_3d'])

                # --- CORRECTED PRINT STATEMENT ---
                if i == 0 and dist.get_rank() == 0:
                    # Use .item() to convert the tensor to a Python float
                    print(f"\n[DEBUG] Negative Sample | Max Confidence: {max_confidence.item():.4f}")

                if max_confidence < self.rejection_thresh:
                    self.tn += 1
                else:
                    self.fp += 1

    def synchronize_between_processes(self):
        """在分布式训练中同步所有进程的计数器"""
        stats = torch.tensor([self.tp, self.tn, self.fp, self.fn, self.pos_count, self.neg_count],
                             device='cuda')
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        self.tp, self.tn, self.fp, self.fn, self.pos_count, self.neg_count = stats.tolist()

    def compute_f1(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    def print_stats(self):
        """打印最终的评估结果"""
        total_samples = self.pos_count + self.neg_count
        
        loc_accuracy = self.tp / (self.pos_count + 1e-8)
        rej_accuracy = self.tn / (self.neg_count + 1e-8)
        overall_accuracy = (self.tp + self.tn) / (total_samples + 1e-8)

        print("\n" + "="*50)
        print(" Comprehensive Evaluation Results".center(50, "="))
        print("="*50)
        print(f"  IoU Threshold for Localization: {self.iou_thresh}")
        print(f"  Confidence Threshold for Rejection: {self.rejection_thresh}\n")
        
        print(f"  - Localization on Positive Samples:")
        print(f"    - Correctly Localized (TP): {self.tp}")
        print(f"    - Failed to Localize (FN):  {self.fn}")
        print(f"    - Localization Accuracy (TP / Positives): {loc_accuracy:.4f}\n")

        print(f"  - Rejection on Negative Samples:")
        print(f"    - Correctly Rejected (TN): {self.tn}")
        print(f"    - Incorrectly Localized (FP): {self.fp}")
        print(f"    - Rejection Accuracy (TN / Negatives): {rej_accuracy:.4f}\n")

        print(f"  - Overall Performance:")
        print(f"    - Total Positive Samples: {self.pos_count}")
        print(f"    - Total Negative Samples: {self.neg_count}")
        print(f"    - Overall Accuracy ((TP + TN) / Total): {overall_accuracy:.4f}")
        print("="*50 + "\n")