# Copyright (c) OpenMMLab. All rights reserved.
"""
DynamicSoftLabelAssigner with area-refine soft label calibration.

Core idea (from USAA):
    For small objects, the IoU used as soft label is boosted because
    small objects are inherently harder to match with high IoU values.
    The calibration is controlled by a per-GT reliability ratio rho:

        rho = area / (area + r_ref^2)

    - Small objects (area ≪ r_ref²): rho → 0, soft-label IoU boosted toward 1
    - Large objects (area ≫ r_ref²): rho → 1, soft-label IoU unchanged

Two calibration modes:
    - 'pow':   soft_iou = iou^rho
    - 'add_1': soft_iou = iou + (1-rho) * iou * (1-iou)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .dynamic_soft_label_assigner import (INF, EPS, DynamicSoftLabelAssigner,
                                           center_of_mass)


@TASK_UTILS.register_module()
class DynamicSoftLabelAssignerAreaRefine(DynamicSoftLabelAssigner):
    """DynamicSoftLabelAssigner with per-GT area-refine soft label calibration.

    Inherits the full DynamicSoftLabelAssigner pipeline and only modifies
    the soft-label construction: the IoU used as soft-label target ceiling
    is calibrated by a per-GT reliability ratio ``rho`` so that small
    objects receive a lifted soft-label ceiling.

    Args:
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions for dynamic k calculation.
            Defaults to 13.
        iou_weight (float): Scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
        r_ref (float): Reference object side length (pixels). The
            transition point where rho = 0.5, i.e. area = r_ref^2.
            Defaults to 32.0.
        calibrate_mode (str): Calibration function.
            ``'pow'`` → ``iou^rho``,
            ``'add_1'`` → ``iou + (1-rho) * iou * (1-iou)``.
            Defaults to ``'pow'``.
    """

    def __init__(self,
                 soft_center_radius: float = 3.0,
                 topk: int = 13,
                 iou_weight: float = 3.0,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'pow') -> None:
        super().__init__(
            soft_center_radius=soft_center_radius,
            topk=topk,
            iou_weight=iou_weight,
            iou_calculator=iou_calculator)
        self.r_ref = r_ref
        self.calibrate_mode = calibrate_mode
        assert calibrate_mode in ('pow', 'add_1'), \
            f'calibrate_mode must be "pow" or "add_1", got {calibrate_mode!r}'

    def _compute_rho(self, gt_bboxes: Tensor) -> Tensor:
        """Compute per-GT reliability ratio from object area.

        rho = area / (area + r_ref^2)

        Args:
            gt_bboxes (Tensor): shape (num_gt, 4) in xyxy format.

        Returns:
            Tensor: shape (num_gt,) with values in (0, 1].
        """
        if isinstance(gt_bboxes, BaseBoxes):
            gt_areas = gt_bboxes.areas
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_areas = gt_w * gt_h

        r_sq = gt_areas
        r_ref_sq = self.r_ref ** 2
        rho = r_sq / (r_sq + r_ref_sq)  # (num_gt,)
        return rho

    def _calibrate_iou(self, pairwise_ious: Tensor,
                       rho: Tensor) -> Tensor:
        """Apply area-refine calibration to pairwise IoUs.

        Args:
            pairwise_ious (Tensor): shape (num_valid, num_gt).
            rho (Tensor): shape (num_gt,). Per-GT reliability ratio.

        Returns:
            Tensor: calibrated IoUs, same shape as pairwise_ious.
        """
        # rho: (num_gt,) → (1, num_gt) for broadcasting
        rho = rho[None, :]  # (1, num_gt)

        if self.calibrate_mode == 'pow':
            # Small objects (rho→0): iou^0 → 1 → soft label ceiling lifted
            # Large objects (rho→1): iou^1 → iou → unchanged
            calibrated = pairwise_ious.pow(rho)
        elif self.calibrate_mode == 'add_1':
            # iou + (1-rho) * iou * (1-iou)
            # Small objects: adds a large boost
            # Large objects: adds almost nothing
            calibrated = pairwise_ious + \
                (1.0 - rho) * pairwise_ious * (1.0 - pairwise_ious)
        else:
            calibrated = pairwise_ious
        return calibrated

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors with area-refine soft label calibration.

        Same pipeline as DynamicSoftLabelAssigner, except the IoU used
        for constructing the soft label is calibrated by per-GT area.

        Args:
            pred_instances (:obj:`InstanceData`): Model predictions.
            gt_instances (:obj:`InstanceData`): Ground truth annotations.
            gt_instances_ignore (:obj:`InstanceData`, optional): Ignored GTs.

        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1, dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # ---- in-gt filtering (same as parent) ----
        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            lt_ = prior_center[:, None] - gt_bboxes[:, :2]
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0

        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1, dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # ---- gt center & center prior (same as parent) ----
        if hasattr(gt_instances, 'masks'):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        elif isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # ---- IoU computation (same as parent) ----
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        # ★ area-refine: calibrate IoU for soft label only ★
        rho = self._compute_rho(gt_bboxes)                     # (num_gt,)
        soft_label_ious = self._calibrate_iou(pairwise_ious, rho)  # (num_valid, num_gt)

        # ---- soft classification cost (using calibrated IoU) ----
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))
        valid_pred_scores_exp = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * soft_label_ious[..., None]
        scale_factor = soft_label - valid_pred_scores_exp.sigmoid()
        soft_cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores_exp, soft_label,
            reduction='none') * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        # ---- cost matrix & dynamic k matching (same as parent) ----
        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)

        # ---- convert to AssignResult format ----
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
