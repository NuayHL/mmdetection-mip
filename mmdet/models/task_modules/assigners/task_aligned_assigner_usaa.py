# Copyright (c) OpenMMLab. All rights reserved.
"""
TaskAlignedAssignerUSAA — a faithful two-stage (RCNN) port of the Ultralytics
``TaskAlignedAssigner_dyab_dmetric_dscale[_RefineArea]`` (see
``Ultralytics/ultralytics/utils/mla_usaa.py``).

Unlike :class:`DynamicSoftLabelAssignerDScaleDYAB` (which is a SimOTA/DSL-family
*additive-cost* assigner with dyab + area-refine bolted on), this class keeps
the genuine TAL machinery:

    1. Multiplicative align metric          t = s^α × u^β
       (NOT the additive ``α·cls_cost + β·(−log iou) + center`` of DSL).

    2. Fixed top-k candidate selection on t  (NOT dynamic-k by Σ IoU).

    3. TAL soft label normalization::

           target = align_for_score · pos_overlap_cal / pos_align_metric

       where ``pos_*`` are the per-GT maxima over its positives. The
       best-aligned anchor for a GT receives a target equal to its
       (calibrated) overlap; weaker anchors are scaled down proportionally.
       This normalized value — NOT a raw IoU — is what is delivered to the
       classification loss (QualityFocalLoss) via ``AssignResult.max_overlaps``.

    4. dmetric (SimD): three independently-configurable metrics for the three
       assignment roles — ``overlap`` (GT ownership + soft-label ceiling),
       ``align`` (top-k ranking), ``score`` (soft-label shape). Any mode of
       :class:`BboxDistanceMetric` is allowed (``iou``/``giou``/``nwd``/``kl``
       /``wd``/``dotd`` …), so the RCNN soft-target *currency* can be aligned
       with an NWD/KLD RPN (RFLA / NWD-RKA) when desired.

    5. dscale: size-dependent candidate-region expansion (small objects admit
       more candidate proposals).

    6. area-refine: per-GT reliability ratio ``ρ = area / (area + r_ref²)``
       lifts the soft-label ceiling for small objects (``pow`` or ``add_1``).

    7. dyab: pluggable dynamic (α, β). Reuses ``DYAB_REGISTRY`` from
       :mod:`.dynamic_soft_label_assigner_dscale_dyab`.

Why this combines with RFLA / NWD-RKA (1+1 ≥ 2):
    RFLA / NWD-RKA only change the *RPN proposal distribution* — i.e. they
    widen / re-shape the candidate pool entering the RCNN. They do NOT
    guarantee that a proposal they surfaced is actually a good positive; that
    is decided here, dynamically, from prediction quality (``s^α × u^β``). The
    soft label is properly normalized to the overlap range, so a tiny-object
    proposal with mediocre overlap is no longer mislabeled as low-confidence
    (the failure mode of the additive DSL port).

Example
-------
    assigner=dict(
        type='TaskAlignedAssignerUSAA',
        topk=13,
        iou_calculator=dict(type='BboxDistanceMetric'),
        overlap_mode='iou', align_mode='iou', score_mode='iou',
        scale_ratio=1.0, expansion_type='size_dependent', expansion_r_ref=32.0,
        r_ref=64.0, calibrate_mode='add_1', lambda_refine=1.0,
        dyab_type='DyabDSL',
        dyab_kwargs=dict(alpha_base=1.0, beta_base=6.0,
                         delta_alpha=0.5, delta_beta=3.0, r_ref=64.0))
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .dynamic_soft_label_assigner_dscale_dyab import DYAB_REGISTRY

INF = 100000000
EPS = 1.0e-7

# IoU-family modes computed inline (incl. CIoU/DIoU — the USAA default). All
# other modes ('nwd'/'kl'/'wd'/'dotd'/...) are delegated to ``iou_calculator``.
_IOU_FAMILY = ('iou', 'giou', 'diou', 'ciou')


@TASK_UTILS.register_module()
class TaskAlignedAssignerUSAA(BaseAssigner):
    """Task-Aligned assigner (USAA) for two-stage RCNN heads.

    Args:
        topk (int): Number of candidate proposals selected per GT (fixed
            top-k on the align metric ``s^α × u^β``). Defaults to 10 (USAA
            standard).
        iou_calculator (ConfigType): Pairwise metric calculator used for the
            non-IoU-family modes ('nwd'/'kl'/'wd'/'dotd'). Must accept a
            ``mode=`` kwarg. Defaults to ``dict(type='BboxDistanceMetric')``.
            IoU-family modes (iou/giou/diou/ciou) are computed inline.
        overlap_mode (str): Metric mode for GT ownership and the soft-label
            ceiling. Defaults to ``'ciou'`` (USAA standard).
        align_mode (str): Metric mode for top-k candidate ranking.
            Defaults to ``'ciou'``.
        score_mode (str): Metric mode shaping the soft label across anchors.
            Defaults to ``'ciou'``.
        scale_ratio (float): Max candidate-expansion ratio. Per-GT margin is
            ``scale_ratio × stride`` (static) or
            ``scale_ratio × (1−ρ) × stride`` (size_dependent). ``0`` disables
            expansion (strict in-gt containment). Defaults to 1.0.
        expansion_type (str): ``'static'`` (USAA standard) | ``'size_dependent'``.
        expansion_r_ref (float): Reference side length for size-dependent
            expansion (area = r_ref² → ρ = 0.5). Defaults to 32.0.
        r_ref (float): Area-refine reference side length for the soft-label
            ceiling calibration. Defaults to 32.0 (USAA standard ``r_ref``).
        calibrate_mode (str): ``'pow'`` → ``overlap^ρ``,
            ``'add_1'`` → ``overlap + λ·(1−ρ)·overlap·(1−overlap)``,
            ``'none'`` → no calibration. Defaults to ``'add_1'``.
        lambda_refine (float): Strength of the ``add_1`` calibration.
            Defaults to 1.0.
        dyab_type (str): Key in ``DYAB_REGISTRY``. Defaults to
            ``'DyabCalibrationAware'`` (USAA standard): small objects get
            ``α↓, β↑`` (less cls weight, sharper IoU ranking) to compensate
            for the area-refine soft-label ceiling boost. (Do NOT confuse with
            ``DyabDSL``, which moves α/β the OPPOSITE way for the additive-cost
            DSL family.)
        dyab_kwargs (dict): Keyword arguments for the Dyab subclass.
        eps (float): Numerical-stability epsilon. Defaults to 1e-7.
    """

    def __init__(self,
                 topk: int = 10,
                 iou_calculator: ConfigType = dict(type='BboxDistanceMetric'),
                 overlap_mode: str = 'ciou',
                 align_mode: str = 'ciou',
                 score_mode: str = 'ciou',
                 scale_ratio: float = 1.0,
                 expansion_type: str = 'static',
                 expansion_r_ref: float = 32.0,
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'add_1',
                 lambda_refine: float = 1.0,
                 dyab_type: str = 'DyabCalibrationAware',
                 dyab_kwargs: Optional[dict] = None,
                 eps: float = 1.0e-7) -> None:
        self.topk = topk
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.overlap_mode = overlap_mode
        self.align_mode = align_mode
        self.score_mode = score_mode

        self.scale_ratio = scale_ratio
        self.expansion_type = expansion_type
        assert expansion_type in ('static', 'size_dependent'), \
            f'Invalid expansion_type: {expansion_type}'
        self.expansion_r_ref_sq = expansion_r_ref ** 2

        self.r_ref_sq = r_ref ** 2
        self.calibrate_mode = calibrate_mode
        assert calibrate_mode in ('pow', 'add_1', 'none'), \
            f'Invalid calibrate_mode: {calibrate_mode}'
        self.lambda_refine = lambda_refine

        dyab_kwargs = dyab_kwargs or {}
        dyab_cls = DYAB_REGISTRY[dyab_type] if isinstance(dyab_type, str) \
            else dyab_type
        self.dyab = dyab_cls(**dyab_kwargs)

        self.eps = eps

    # ── dscale: candidate filtering ───────────────────────────────────────

    def _filter_candidates_in_gts(self, priors: Tensor,
                                   gt_bboxes: Tensor) -> Tensor:
        """Size-dependent candidate mask, shape (num_priors, num_gt).

        ``priors[:, :2]`` is the centre, ``priors[:, 2]`` the per-prior
        stride (set to ``min(w, h)`` by ``DynAssignRoIHead`` point format).
        """
        prior_center = priors[:, :2]
        strides_all = priors[:, 2]

        if isinstance(gt_bboxes, BaseBoxes):
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            else:
                return gt_bboxes.find_inside_points(prior_center)

        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
        deltas = torch.cat([lt_, rb_], dim=-1)

        if self.scale_ratio == 0.0:
            return deltas.min(dim=-1).values > 0

        if self.expansion_type == 'static':
            per_gt_scale = gt_bboxes.new_full((gt_bboxes.shape[0], ),
                                              self.scale_ratio)
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_area = gt_w * gt_h
            rho = gt_area / (gt_area + self.expansion_r_ref_sq)
            per_gt_scale = self.scale_ratio * (1.0 - rho)

        margin = per_gt_scale[None, :] * strides_all[:, None]
        return deltas.min(dim=-1).values > (-margin)

    # ── dmetric helpers ───────────────────────────────────────────────────

    def _metric(self, boxes: Tensor, gt_bboxes: Tensor, mode: str) -> Tensor:
        """Pairwise metric, shape (num_boxes, num_gt), clamped to >= 0.

        IoU-family modes (incl. CIoU/DIoU — the USAA default) are computed
        inline; distance metrics ('nwd'/'kl'/'wd'/...) go through the
        configured ``iou_calculator`` (BboxDistanceMetric).
        """
        if mode in _IOU_FAMILY:
            m = self._iou_family(boxes, gt_bboxes, mode)
        else:
            m = self.iou_calculator(boxes, gt_bboxes, mode=mode)
        return m.clamp(min=0)

    def _iou_family(self, boxes1: Tensor, boxes2: Tensor,
                    mode: str) -> Tensor:
        """Pairwise IoU/GIoU/DIoU/CIoU, shape (N, M). Matches the Ultralytics
        ``bbox_iou(CIoU=True)`` used as the USAA dmetric (clamping to >= 0 is
        done by the caller)."""
        boxes1 = get_box_tensor(boxes1)
        boxes2 = get_box_tensor(boxes2)
        n, m = boxes1.size(0), boxes2.size(0)
        if n * m == 0:
            return boxes1.new_zeros((n, m))

        w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        area1, area2 = w1 * h1, w2 * h2

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2[None, :] - overlap + self.eps
        ious = overlap / union
        if mode == 'iou':
            return ious

        enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
        enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
        enc_wh = (enc_rb - enc_lt).clamp(min=0)
        if mode == 'giou':
            enc_area = enc_wh[..., 0] * enc_wh[..., 1] + self.eps
            return ious - (enc_area - union) / enc_area

        # center distance² and enclosing diagonal² (DIoU / CIoU)
        cx1 = (boxes1[:, 0] + boxes1[:, 2]) * 0.5
        cy1 = (boxes1[:, 1] + boxes1[:, 3]) * 0.5
        cx2 = (boxes2[:, 0] + boxes2[:, 2]) * 0.5
        cy2 = (boxes2[:, 1] + boxes2[:, 3]) * 0.5
        rho2 = (cx1[:, None] - cx2[None, :]) ** 2 + \
               (cy1[:, None] - cy2[None, :]) ** 2
        c2 = enc_wh[..., 0] ** 2 + enc_wh[..., 1] ** 2 + self.eps
        diou = ious - rho2 / c2
        if mode == 'diou':
            return diou

        # CIoU aspect-ratio penalty
        factor = 4.0 / (math.pi ** 2)
        v = factor * (torch.atan(w2 / (h2 + self.eps))[None, :] -
                      torch.atan(w1 / (h1 + self.eps))[:, None]) ** 2
        with torch.no_grad():
            alpha_ciou = v / (1 - ious + v + self.eps)
        return diou - alpha_ciou * v

    # ── area-refine helpers ───────────────────────────────────────────────

    def _compute_rho(self, gt_bboxes: Tensor) -> Tensor:
        """Per-GT reliability ratio ``ρ = area / (area + r_ref²)``."""
        if isinstance(gt_bboxes, BaseBoxes):
            gt_areas = gt_bboxes.areas
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_areas = gt_w * gt_h
        return gt_areas / (gt_areas + self.r_ref_sq)

    def _calibrate(self, overlap: Tensor, rho: Tensor) -> Tensor:
        """Lift the per-GT soft-label ceiling for small objects."""
        if self.calibrate_mode == 'pow':
            return overlap.pow(rho)
        if self.calibrate_mode == 'add_1':
            return overlap + self.lambda_refine * (1.0 - rho) * \
                overlap * (1.0 - overlap)
        return overlap

    # ── dyab helper ───────────────────────────────────────────────────────

    def _compute_dyab(self, uncertainty: Optional[Tensor], gt_bboxes: Tensor,
                      num_valid: int, num_gt: int) -> Tuple[Tensor, Tensor]:
        """Per (anchor, GT) α, β, broadcast to shape (num_valid, num_gt)."""
        alpha, beta = self.dyab.compute(
            uncertainty, gt_bboxes, num_gt, num_valid, device=gt_bboxes.device)
        if isinstance(alpha, (int, float)):
            alpha = gt_bboxes.new_full((num_valid, num_gt), float(alpha))
            beta = gt_bboxes.new_full((num_valid, num_gt), float(beta))
        elif alpha.dim() == 1:
            alpha = alpha[None, :].expand(num_valid, num_gt)
            beta = beta[None, :].expand(num_valid, num_gt)
        return alpha, beta

    # ── selection helpers ─────────────────────────────────────────────────

    def _select_topk(self, align_metric: Tensor) -> Tensor:
        """Top-k anchors per GT (column-wise), shape (num_valid, num_gt)."""
        num_valid = align_metric.size(0)
        k = min(self.topk, num_valid)
        _, topk_idx = torch.topk(align_metric, k, dim=0)
        mask = torch.zeros_like(align_metric, dtype=torch.bool)
        mask.scatter_(0, topk_idx, True)
        # drop candidates with zero align metric (e.g. outside-gt padding).
        return mask & (align_metric > 0)

    @staticmethod
    def _resolve_conflicts(mask_pos: Tensor,
                           overlaps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """One anchor → at most one GT (the highest-overlap one).

        Returns (target_gt_idx, fg_mask, mask_pos).
        """
        num_pos_per_anchor = mask_pos.sum(dim=1)
        multi = num_pos_per_anchor > 1
        if multi.any():
            ov = torch.where(mask_pos, overlaps,
                             overlaps.new_full((), -1.0))
            best_gt = ov.argmax(dim=1)
            new_mask = torch.zeros_like(mask_pos)
            new_mask[torch.arange(mask_pos.size(0), device=mask_pos.device),
                     best_gt] = True
            mask_pos = torch.where(multi[:, None], new_mask, mask_pos)
        fg_mask = mask_pos.sum(dim=1) > 0
        target_gt_idx = mask_pos.float().argmax(dim=1)
        return target_gt_idx, fg_mask, mask_pos

    # ── main assign ───────────────────────────────────────────────────────

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores          # (N, C) sigmoid probs
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)
        uncertainty = getattr(pred_instances, 'uncertainty', None)

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ), 0,
                                                   dtype=torch.long)

        def _empty() -> AssignResult:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ), -1,
                                                      dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps,
                                labels=assigned_labels)

        if num_gt == 0 or num_bboxes == 0:
            return _empty()

        # 1. dscale candidate filtering -----------------------------------
        is_in_gts = self._filter_candidates_in_gts(priors, gt_bboxes)
        valid_mask = is_in_gts.sum(dim=1) > 0
        if valid_mask.sum() == 0:
            return _empty()

        valid_decoded = decoded_bboxes[valid_mask]
        valid_scores = pred_scores[valid_mask]
        valid_in_gts = is_in_gts[valid_mask]
        num_valid = valid_decoded.size(0)
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)

        # 2. dmetric: three roles -----------------------------------------
        overlaps = self._metric(valid_decoded, gt_bboxes, self.overlap_mode)
        align_iou = self._metric(valid_decoded, gt_bboxes, self.align_mode)
        score_iou = self._metric(valid_decoded, gt_bboxes, self.score_mode)

        # 3. cls score for each (anchor, GT-class) ------------------------
        # valid_scores: (num_valid, C); gt_labels: (num_gt,) -> (num_valid, num_gt)
        bbox_scores = valid_scores[:, gt_labels.long()].clamp(min=self.eps,
                                                              max=1.0)

        # 4. dyab dynamic (α, β) ------------------------------------------
        alpha, beta = self._compute_dyab(
            uncertainty[valid_mask] if uncertainty is not None else None,
            gt_bboxes, num_valid, num_gt)

        # 5. multiplicative align metric  t = s^α × u^β -------------------
        s_part = bbox_scores.pow(alpha)
        align_for_align = s_part * align_iou.pow(beta) * valid_in_gts
        align_for_score = s_part * score_iou.pow(beta) * valid_in_gts

        # 6. fixed top-k selection + conflict resolution ------------------
        mask_pos = self._select_topk(align_for_align) & valid_in_gts
        target_gt_idx, fg_mask, mask_pos = self._resolve_conflicts(
            mask_pos, overlaps)

        if fg_mask.sum() == 0:
            return _empty()

        # 7. TAL soft-label normalization ---------------------------------
        rho = self._compute_rho(gt_bboxes)                         # (num_gt,)
        align_pos = align_for_score * mask_pos
        pos_align_metric = align_pos.amax(dim=0)                   # (num_gt,)
        pos_overlap = (overlaps * mask_pos).amax(dim=0)            # (num_gt,)
        pos_overlap_cal = self._calibrate(pos_overlap, rho)        # (num_gt,)

        norm = align_for_score * pos_overlap_cal[None, :] / \
            (pos_align_metric[None, :] + self.eps)
        norm = norm * mask_pos
        soft_target = norm.amax(dim=1)                             # (num_valid,)

        # 8. write AssignResult (TAL soft label -> max_overlaps) ----------
        fg_full = valid_idx[fg_mask]
        assigned_gt_inds[fg_full] = target_gt_idx[fg_mask] + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[fg_full] = gt_labels[target_gt_idx[fg_mask]].long()

        max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
        max_overlaps[fg_full] = soft_target[fg_mask]
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps,
                            labels=assigned_labels)
