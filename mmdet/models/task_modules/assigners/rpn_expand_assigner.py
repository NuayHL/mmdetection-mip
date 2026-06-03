# Copyright (c) OpenMMLab. All rights reserved.
"""
RPN assigner with candidate-region expansion for small objects.

Extends MaxIoUAssigner with expansion logic: anchors whose centres fall
outside a GT box by up to ``scale_ratio × stride`` pixels are still
considered as candidates.  The effective stride is estimated from the
anchor's shorter side: ``stride ≈ sqrt(w×h) / base_scale``.
"""

from typing import Optional

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner


@TASK_UTILS.register_module()
class RPNExpandAssigner(MaxIoUAssigner):
    """RPN assigner with expansion for small-object recall.

    Wraps MaxIoUAssigner.  Before computing IoU-based assignment, anchors
    that are too far from any GT box (according to the expansion margin)
    are masked out, effectively expanding the candidate pool for small GTs.

    Args:
        pos_iou_thr (float): IoU threshold for positive anchors.
        neg_iou_thr (float): IoU threshold for negative anchors.
        min_pos_iou (float): Minimum IoU for positive anchors.
        scale_ratio (float): Expansion margin multiplier.  An anchor whose
            centre is within ``scale_ratio × stride`` pixels outside a GT
            box is still a candidate.  Defaults to 1.0.
        base_scale (int): Base scale of the anchor generator, used to
            estimate stride from anchor size.  Defaults to 8 (standard
            for mmdet AnchorGenerator with scales=[8]).
        expansion_r_ref (float): Reference size for size-dependent
            expansion (only used when ``expansion_type='size_dependent'``).
        expansion_type (str): ``'static'`` (uniform expansion) or
            ``'size_dependent'`` (less expansion for large objects).
            Defaults to ``'static'``.
    """

    def __init__(self,
                 pos_iou_thr: float = 0.7,
                 neg_iou_thr: float = 0.3,
                 min_pos_iou: float = 0.3,
                 scale_ratio: float = 1.0,
                 base_scale: int = 8,
                 expansion_r_ref: float = 32.0,
                 expansion_type: str = 'static',
                 **kwargs) -> None:
        super().__init__(
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
            min_pos_iou=min_pos_iou,
            **kwargs)
        self.scale_ratio = scale_ratio
        self.base_scale = base_scale
        self.expansion_r_ref_sq = expansion_r_ref ** 2
        self.expansion_type = expansion_type
        assert expansion_type in ('static', 'size_dependent'), \
            f'Invalid expansion_type: {expansion_type!r}'

    def _estimate_stride(self, anchors: Tensor) -> Tensor:
        """Estimate feature stride from anchor dimensions.

        For standard RPN anchors with base_scale:
            stride ≈ sqrt(w × h) / base_scale

        Args:
            anchors (Tensor): (N, 4) xyxy.

        Returns:
            Tensor: (N,) estimated stride per anchor.
        """
        w = anchors[:, 2] - anchors[:, 0]
        h = anchors[:, 3] - anchors[:, 1]
        area = (w * h).clamp(min=1.0)
        stride = area.sqrt() / self.base_scale
        return stride

    def _get_expansion_mask(self, anchors: Tensor,
                            gt_bboxes: Tensor) -> Tensor:
        """Compute which (anchor, GT) pairs pass the expansion filter.

        An anchor is a candidate for a GT if its centre is no more than
        ``scale_ratio × stride`` pixels outside the GT box.

        Args:
            anchors (Tensor): (N, 4) xyxy.
            gt_bboxes (Tensor): (M, 4) xyxy.

        Returns:
            Tensor[bool]: (N, M) mask.
        """
        # anchor centres and estimated strides
        center_x = (anchors[:, 0] + anchors[:, 2]) / 2.0     # (N,)
        center_y = (anchors[:, 1] + anchors[:, 3]) / 2.0     # (N,)
        strides = self._estimate_stride(anchors)              # (N,)

        # deltas: (N, M, 4) = [left, top, right, bottom]
        lt_x = center_x[:, None] - gt_bboxes[None, :, 0]     # (N, M)
        lt_y = center_y[:, None] - gt_bboxes[None, :, 1]     # (N, M)
        rb_x = gt_bboxes[None, :, 2] - center_x[:, None]     # (N, M)
        rb_y = gt_bboxes[None, :, 3] - center_y[:, None]     # (N, M)
        deltas_min = torch.stack([lt_x, lt_y, rb_x, rb_y], dim=-1).min(dim=-1).values

        if self.scale_ratio == 0.0:
            return deltas_min > 0

        if self.expansion_type == 'static':
            per_gt_scale = self.scale_ratio
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_area = gt_w * gt_h
            rho = gt_area / (gt_area + self.expansion_r_ref_sq)
            per_gt_scale = self.scale_ratio * (1.0 - rho)          # (M,)

        margin = per_gt_scale * strides                            # (N,) or (N, M)
        if margin.dim() == 1:
            margin = margin[:, None]                               # (N, 1) → broadcast to (N, M)
        return deltas_min > (-margin)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign with expansion filtering.

        1. Compute expansion mask → which anchors are candidates per GT.
        2. Compute IoU between all anchors and GTs.
        3. Zero out IoU for non-candidate (anchor, GT) pairs.
        4. Apply standard MaxIoU thresholds on the masked IoU matrix.
        """
        anchors = pred_instances.priors         # (N, 4) xyxy
        gt_bboxes = gt_instances.bboxes          # (M, 4)
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)
        num_anchors = anchors.size(0)

        # Default assignment
        assigned_gt_inds = anchors.new_full((num_anchors,), 0, dtype=torch.long)
        if num_gt == 0 or num_anchors == 0:
            max_overlaps = anchors.new_zeros((num_anchors,))
            assigned_labels = anchors.new_full((num_anchors,), -1, dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # Compute overlaps
        overlaps = self.iou_calculator(anchors, gt_bboxes)      # (N, M)

        # Apply expansion mask: zero out non-candidates
        if self.scale_ratio > 0.0:
            expand_mask = self._get_expansion_mask(anchors, gt_bboxes)
            overlaps = overlaps * expand_mask.float()

        # Standard MaxIoU assignment on masked overlaps
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result
