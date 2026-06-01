# Copyright (c) OpenMMLab. All rights reserved.
"""
DynamicSoftLabelAssigner with candidate expansion + dyab + area-refine soft label.

Three enhancements over the vanilla DynamicSoftLabelAssigner:

    expansion (dscale)
        The strict "centre must lie inside the GT box" filter is relaxed:
        centres may fall outside by up to ``scale_ratio × stride`` pixels.
        ``scale_ratio=0`` recovers the original strict containment.

    dyab
        Dynamic α / β weights for the classification cost and the IoU cost
        in the matching cost matrix::

            cost = α · soft_cls_cost + β · iou_cost + soft_center_prior

        Configure via ``dyab_type`` + ``dyab_kwargs``.

    area-refine soft label (inherited from DynamicSoftLabelAssignerAreaRefine)
        The IoU ceiling of the soft label is lifted for small objects via a
        per-GT reliability ratio ρ = area / (area + r_ref²).

Example YAML
------------
    assigner=dict(
        type='DynamicSoftLabelAssignerDScaleDYAB',
        soft_center_radius=3.0,
        topk=13,
        iou_weight=3.0,
        r_ref=32.0,
        calibrate_mode='add_1',
        scale_ratio=1.0,             # expand by 1× stride
        dyab_type='DyabCalibrationAware',
        dyab_kwargs=dict(alpha_base=1.0, beta_base=4.0,
                         delta_alpha=0.5, delta_beta=2.0, r_ref=64.0),
    )
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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
from .dynamic_soft_label_assigner_area_refine import \
    DynamicSoftLabelAssignerAreaRefine


# =============================================================================
# DyabBase / concrete implementations — dynamic α / β computation
# =============================================================================

class DyabBase:
    """Abstract base for dynamic (α, β) computation.

    Subclasses receive GT geometry (and optionally per-anchor uncertainty)
    and return per (anchor, GT) α and β tensors used to weight the cost matrix:

        cost = α · soft_cls_cost + β · iou_cost + soft_center_prior

    High α → classification cost dominates → more selective on cls quality.
    High β → IoU cost dominates → more selective on localisation quality.
    """

    def compute(self, uncertainty, gt_bboxes, num_gt, na, device):
        raise NotImplementedError


class FixedAB(DyabBase):
    """Fixed α / β — backward-compatible with vanilla DSL."""

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def compute(self, uncertainty, gt_bboxes, num_gt, na, device):
        return self.alpha, self.beta


class DyabBudgetShift(DyabBase):
    """Budget-preserving shift: α + β = const.

    Only depends on GT area — no per-anchor uncertainty needed.

    α_i = α₀ + Δ · (1 - ρ_i)      (small obj → higher cls weight)
    β_i = β₀ - Δ · (1 - ρ_i)      (small obj → lower IoU weight)

    ρ_i = area / (area + r_ref²)   (small obj → ρ → 0)

    YAML::

        dyab_type: DyabBudgetShift
        dyab_kwargs: {alpha_base: 1.0, beta_base: 4.0, delta: 1.5, r_ref: 32.0}
    """

    def __init__(self, alpha_base=1.0, beta_base=4.0,
                 delta=1.5, r_ref=32.0):
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        self.delta = delta
        self.r_ref_sq = r_ref ** 2

    def compute(self, uncertainty, gt_bboxes, num_gt, na, device):
        if isinstance(gt_bboxes, BaseBoxes):
            gt_areas = gt_bboxes.areas
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_areas = gt_w * gt_h
        r_sq = gt_areas                         # (num_gt,)
        rho = r_sq / (r_sq + self.r_ref_sq)     # ∈ (0, 1]
        shift = self.delta * (1.0 - rho)         # (num_gt,)
        alpha = self.alpha_base + shift
        beta = self.beta_base - shift
        return alpha, beta


class DyabLinearFusion(DyabBase):
    """Linear fusion of per-anchor uncertainty and GT area → dynamic α / β.

    Difficulty score:
        k = λ · u_norm + (1 - λ) · (1 - area_score)
    k = 1 → hardest (small + uncertain) → α_hard, β_hard
    k = 0 → easiest  (large + confident) → α_easy, β_easy

    YAML::

        dyab_type: DyabLinearFusion
        dyab_kwargs:
          lambda_fusion: 0.6  uncertainty_tau: 2.0
          scale_min: 16  scale_max: 64
          alpha_easy: 0.5  alpha_hard: 1.2
          beta_easy: 6.0  beta_hard: 3.0
    """

    def __init__(self, lambda_fusion=0.6, uncertainty_tau=2.0,
                 scale_min=16, scale_max=64,
                 alpha_easy=0.5, alpha_hard=1.2,
                 beta_easy=6.0, beta_hard=3.0):
        self.lambda_fusion = lambda_fusion
        self.uncertainty_tau = uncertainty_tau
        self.log_scale_min = math.log(scale_min)
        self.log_scale_max = math.log(scale_max)
        self.alpha_easy = alpha_easy
        self.alpha_hard = alpha_hard
        self.beta_easy = beta_easy
        self.beta_hard = beta_hard

    def compute(self, uncertainty, gt_bboxes, num_gt, na, device):
        if isinstance(gt_bboxes, BaseBoxes):
            gt_areas = gt_bboxes.areas
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_areas = gt_w * gt_h
        gt_scale = torch.sqrt(torch.clamp(gt_areas, min=1.0))
        area_score = ((gt_scale.log() - self.log_scale_min) /
                      (self.log_scale_max - self.log_scale_min)).clamp(0.0, 1.0)
        if uncertainty is not None and uncertainty.numel() > 0:
            u_norm = torch.tanh(uncertainty / self.uncertainty_tau)
        else:
            u_norm = torch.zeros(na, device=device)
        u_mean = u_norm.mean()
        k = self.lambda_fusion * u_mean + (1 - self.lambda_fusion) * (1 - area_score)
        dynamic_alpha = self.alpha_easy + (self.alpha_hard - self.alpha_easy) * k
        dynamic_beta = self.beta_easy + (self.beta_hard - self.beta_easy) * k
        return dynamic_alpha, dynamic_beta


class DyabCalibrationAware(DyabBase):
    """Calibration-aware α / β — complements area-refine soft label.

    α_i = α₀ - Δα · (1 - ρ_i)
    β_i = β₀ + Δβ · (1 - ρ_i)

    When the soft-label ceiling is boosted for small objects (via area-refine
    calibration), the cls scores become compressed.  This compensates by
    increasing β (sharper IoU ranking) and reducing α for small objects.

    YAML::

        dyab_type: DyabCalibrationAware
        dyab_kwargs: {alpha_base: 1.0, beta_base: 4.0,
                      delta_alpha: 0.5, delta_beta: 2.0, r_ref: 64.0}
    """

    def __init__(self, alpha_base=1.0, beta_base=4.0,
                 delta_alpha=0.5, delta_beta=2.0, r_ref=32.0):
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        self.delta_alpha = delta_alpha
        self.delta_beta = delta_beta
        self.r_ref_sq = r_ref ** 2

    def compute(self, uncertainty, gt_bboxes, num_gt, na, device):
        if isinstance(gt_bboxes, BaseBoxes):
            gt_areas = gt_bboxes.areas
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_areas = gt_w * gt_h
        r_sq = gt_areas
        rho = r_sq / (r_sq + self.r_ref_sq)
        c = 1.0 - rho                                # calibration strength
        alpha = self.alpha_base - self.delta_alpha * c
        beta = self.beta_base + self.delta_beta * c
        return alpha, beta


DYAB_REGISTRY = {
    'FixedAB': FixedAB,
    'DyabBudgetShift': DyabBudgetShift,
    'DyabLinearFusion': DyabLinearFusion,
    'DyabCalibrationAware': DyabCalibrationAware,
}


# =============================================================================
# DynamicSoftLabelAssignerDScaleDYAB
# =============================================================================

@TASK_UTILS.register_module()
class DynamicSoftLabelAssignerDScaleDYAB(DynamicSoftLabelAssignerAreaRefine):
    """DSL assigner with candidate expansion + dyab cost weighting.

    Inherits area-refine soft-label calibration from
    :class:`DynamicSoftLabelAssignerAreaRefine`.

    **expansion**
        The strict in-gt containment is relaxed: prediction centres may lie
        outside a GT box by up to ``scale_ratio × stride`` pixels::

            deltas.min > -(scale_ratio × stride)

        - ``scale_ratio = 0`` → original strict containment
        - ``scale_ratio = 1.0`` → centre can be up to ``stride`` pixels outside

    **dyab**
        The cost matrix uses dynamic per-(anchor, GT) weights::

            cost = α · soft_cls_cost + β · iou_cost + soft_center_prior

    Args:
        soft_center_radius (float): See :class:`DynamicSoftLabelAssigner`.
        topk (int): See :class:`DynamicSoftLabelAssigner`.
        iou_weight (float): Base scale factor for iou_cost.
        iou_calculator (ConfigType): See :class:`DynamicSoftLabelAssigner`.
        r_ref (float): Area-refine reference size.
        calibrate_mode (str): Area-refine calibration mode (``'pow'`` or ``'add_1'``).
        scale_ratio (float): Candidate expansion ratio.  The expansion margin
            is ``scale_ratio × stride``.  Defaults to 1.0.
        dyab_type (str): Key in ``DYAB_REGISTRY``.
        dyab_kwargs (dict): Keyword arguments for the Dyab subclass.
    """

    def __init__(self,
                 soft_center_radius: float = 3.0,
                 topk: int = 13,
                 iou_weight: float = 3.0,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'pow',
                 scale_ratio: float = 1.0,
                 dyab_type: str = 'FixedAB',
                 dyab_kwargs: Optional[dict] = None,
                 ) -> None:
        super().__init__(
            soft_center_radius=soft_center_radius,
            topk=topk,
            iou_weight=iou_weight,
            iou_calculator=iou_calculator,
            r_ref=r_ref,
            calibrate_mode=calibrate_mode)
        self.scale_ratio = scale_ratio

        dyab_kwargs = dyab_kwargs or {}
        dyab_cls = DYAB_REGISTRY[dyab_type] if isinstance(dyab_type, str) else dyab_type
        self.dyab = dyab_cls(**dyab_kwargs)

    # ── expansion: candidate filtering ────────────────────────────────────

    def _filter_candidates_in_gts(self, priors: Tensor,
                                   gt_bboxes: Tensor) -> Tensor:
        """Select anchor centres that are candidates for each GT.

        Expansion: centre may be outside the GT box by up to
        ``scale_ratio × stride`` pixels::

            margin = scale_ratio × stride
            deltas.min > -margin

        ``scale_ratio = 0`` recovers the original strict containment.

        Args:
            priors (Tensor): (num_bboxes, ≥3) — ``priors[:, :2]`` centre,
                ``priors[:, 2]`` stride.
            gt_bboxes (Tensor or BaseBoxes): (num_gt, 4) xyxy.

        Returns:
            Tensor[bool]: (num_bboxes, num_gt) candidate mask.
        """
        prior_center = priors[:, :2]             # (num_bboxes, 2)
        strides_all = priors[:, 2]               # (num_bboxes,)

        # Unwrap BaseBoxes → Tensor for expansion; rotated boxes fall back
        # to strict containment (expansion margin doesn't make geometric
        # sense for rotated geometry).
        if isinstance(gt_bboxes, BaseBoxes):
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            else:
                return gt_bboxes.find_inside_points(prior_center)

        lt_ = prior_center[:, None] - gt_bboxes[:, :2]        # (n, m, 2)
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]         # (n, m, 2)
        deltas = torch.cat([lt_, rb_], dim=-1)                 # (n, m, 4)

        if self.scale_ratio == 0.0:
            return deltas.min(dim=-1).values > 0

        # expansion: margin = scale_ratio × stride
        margin = self.scale_ratio * strides_all               # (n,)
        is_in_gts = deltas.min(dim=-1).values > (-margin[:, None])
        return is_in_gts

    # ── dyab: dynamic cost weighting ──────────────────────────────────────

    def _compute_dynamic_weights(self, uncertainty: Optional[Tensor],
                                  gt_bboxes: Tensor, num_valid: int,
                                  num_gt: int) -> Tuple[Tensor, Tensor]:
        """Compute per (anchor, GT) α, β weights for the cost matrix."""
        alpha, beta = self.dyab.compute(
            uncertainty, gt_bboxes, num_gt, num_valid,
            device=gt_bboxes.device)
        if isinstance(alpha, (int, float)):
            alpha = torch.full((num_valid, num_gt), float(alpha),
                               device=gt_bboxes.device)
            beta = torch.full((num_valid, num_gt), float(beta),
                              device=gt_bboxes.device)
        elif alpha.dim() == 1:
            alpha = alpha[None, :].expand(num_valid, num_gt)
            beta = beta[None, :].expand(num_valid, num_gt)
        return alpha, beta

    # ── main assign ───────────────────────────────────────────────────────

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors with expansion + dyab + area-refine.

        1. Expansion candidate filtering
        2. Area-refine soft-label IoU calibration
        3. dyab-weighted cost matrix
        4. Dynamic-k matching
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        uncertainty = getattr(pred_instances, 'uncertainty', None)

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

        # ---- expansion candidate filtering ----
        is_in_gts = self._filter_candidates_in_gts(priors, gt_bboxes)
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

        # ---- gt center & soft center prior ----
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

        # ---- IoU computation ----
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        # ---- area-refine: calibrate IoU for soft label ----
        rho = self._compute_rho(gt_bboxes)
        soft_label_ious = self._calibrate_iou(pairwise_ious, rho)

        # ---- dyab: dynamic cost weights ----
        dynamic_alpha, dynamic_beta = self._compute_dynamic_weights(
            uncertainty[valid_mask] if uncertainty is not None else None,
            gt_bboxes, num_valid, num_gt)

        # ---- soft classification cost ----
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

        # ---- dyab-weighted cost matrix ----
        cost_matrix = (dynamic_alpha * soft_cls_cost +
                       dynamic_beta * iou_cost +
                       soft_center_prior)

        # ---- dynamic k matching ----
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)

        # ---- convert to AssignResult ----
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
