# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from mmdet.models.losses.utils import weighted_loss


def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
    ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
    of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
        bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "giou" (generalized
            intersection over union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Defaults to False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Defaults to 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 10, 10],
        >>>     [10, 10, 10, 20, 20, 20],
        >>>     [32, 32, 32, 38, 40, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 20, 20],
        >>>     [0, 10, 10, 10, 19, 20],
        >>>     [10, 10, 10, 20, 20, 20],
        >>> ])
        >>> overlaps = axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 6)
        >>> nonempty = torch.FloatTensor([[0, 0, 0, 10, 9, 10]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

class AxisAlignedBboxOverlaps3D(object):
    """Axis-aligned 3D Overlaps (IoU) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or "giou" (generalized
                intersection over union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Defaults to False.
        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        assert bboxes1.size(-1) == bboxes2.size(-1) == 6
        return axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode,
                                             is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
    
@weighted_loss
def axis_aligned_iou_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of axis aligned bounding
    boxes. Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).
        target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        pred, target, is_aligned=True)
    iou_loss = 1 - axis_aligned_iou
    return iou_loss

@weighted_loss
def axis_aligned_diou_loss(pred, target):
    """Calculate the DIoU loss (1-DIoU) of two sets of axis aligned bounding
    boxes. Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).
        target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        pred, target, is_aligned=True)
    iou_loss = 1 - axis_aligned_iou

    xp1, yp1, zp1, xp2, yp2, zp2 = pred.split(1, dim=-1)
    xt1, yt1, zt1, xt2, yt2, zt2 = target.split(1, dim=-1)

    xpc = (xp1 + xp2) / 2
    ypc = (yp1 + yp2) / 2
    zpc = (zp1 + zp2) / 2
    xtc = (xt1 + xt2) / 2
    ytc = (yt1 + yt2) / 2
    ztc = (zt1 + zt2) / 2
    r2 = (xpc - xtc) ** 2 + (ypc - ytc) ** 2 + (zpc - ztc) ** 2

    x_min = torch.minimum(xp1, xt1)
    x_max = torch.maximum(xp2, xt2)
    y_min = torch.minimum(yp1, yt1)
    y_max = torch.maximum(yp2, yt2)
    z_min = torch.minimum(zp1, zt1)
    z_max = torch.maximum(zp2, zt2)
    c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2

    diou_loss = iou_loss + (r2 / c2)[:, 0]

    return diou_loss

class AxisAlignedIoULoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, mode='iou', reduction='mean', loss_weight=1.0):
        super(AxisAlignedIoULoss, self).__init__()
        self.loss = axis_aligned_iou_loss if mode == 'iou' else axis_aligned_diou_loss
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 6]
                (x1, y1, z1, x2, y2, z2).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
                (x1, y1, z1, x2, y2, z2).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        return self.loss(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction) * self.loss_weight

class AxisAlignedIoULoss2(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, mode='iou', reduction='mean', loss_weight=1.0):
        super(AxisAlignedIoULoss2, self).__init__()
        self.loss = axis_aligned_iou_loss if mode == 'iou' else axis_aligned_diou_loss
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 6]
                (x1, y1, z1, x2, y2, z2).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
                (x1, y1, z1, x2, y2, z2).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        return self.loss(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction) * self.loss_weight