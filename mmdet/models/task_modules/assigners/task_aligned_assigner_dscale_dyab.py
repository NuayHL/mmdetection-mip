# Copyright (c) OpenMMLab. All rights reserved.
"""
TaskAlignedAssignerDScaleDYAB — RTMDet/one-stage port of the ultralytics
``TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea``.

This reproduces, inside mmdetection's per-image assigner interface, the exact
assignment algorithm used to train YOLO in ultralytics with the USAA design:

    alignment metric (+ dyab)
        Multiplicative task-aligned metric ``t = s^α · IoU^β`` with dynamic,
        per-GT (α, β).  Top-k anchors per GT are selected by this metric and
        anchor↔GT conflicts are resolved by raw IoU (``select_highest_overlaps``).
        ``s`` is the predicted *probability* (sigmoid of the cls logit) for the
        GT class.  Configure dynamic α/β via ``dyab_type`` + ``dyab_kwargs``
        (shared registry with the DSL variant — see ``DYAB_REGISTRY``).

    soft label (+ area-refine)
        The soft classification target carried to QualityFocalLoss is the
        normalised alignment metric

            norm = align · pos_overlaps_cal / (pos_align_metrics + eps)

        where ``pos_overlaps_cal`` is the per-GT IoU ceiling after the
        area-refine calibration (small objects get a lifted ceiling via
        ρ = area / (area + r_ref²)).

    dscale
        Size/stride-adaptive candidate region.  ``dscale_func='static'`` relaxes
        containment by a uniform margin ``min(l,t,r,b) > -(stride·scale_ratio)``
        — the same expansion convention as the two-stage port
        ``DynamicSoftLabelAssignerDScaleDYAB`` (``scale_ratio=0`` → strict
        containment, ``scale_ratio=1`` → centre may lie one stride outside).
        This admits more anchors for tiny objects; other ``dscale_func`` options
        make the threshold depend on stride / object size.

**Interface bridge to RTMDet.**
    ``RTMDetHead._get_targets_single`` reads ``AssignResult.max_overlaps`` at the
    positive anchors as the soft label (``assign_metrics``), which is then used
    both as the QualityFocalLoss soft target and the bbox-loss weight — exactly
    the role of ultralytics' ``norm_align_metric``.  We therefore store the
    per-anchor ``norm_align_metric`` in ``max_overlaps``.

**Note on dmetric.**
    The ultralytics class additionally supports three *different* IoU metrics
    (SimD / Hausdorff) for the ownership / ranking / scoring roles.  This port
    uses a single ``iou_calculator`` for all three roles, matching the standard
    TAL configuration.  The default is **CIoU** (``BboxSiM2D`` with
    ``mode='ciou'``) to match the ultralytics default; ``'iou'``, ``'giou'``,
    ``'diou'`` are also available.  The three requested components — alignment
    metric, soft label, dscale — are reproduced exactly.

Example config
--------------
    assigner=dict(
        type='TaskAlignedAssignerDScaleDYAB',
        topk=13,
        dscale_func='static',
        scale_ratio=1.0,
        r_ref=32.0,
        r_ref_type='pow',
        dyab_type='DyabCalibrationAware',
        dyab_kwargs=dict(alpha_base=1.0, beta_base=4.0,
                         delta_alpha=0.5, delta_beta=2.0, r_ref=64.0),
    )
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .dynamic_soft_label_assigner_dscale_dyab import DYAB_REGISTRY

INF = 100000000
EPS = 1.0e-9


# =============================================================================
# DScaleFunctions — stride/size → scale-ratio mapping library (dscale)
# Faithful port of usaa_ultralytics.tal.DScaleFunctions.
# =============================================================================

class DScaleFunctions:
    """Pure-static library of stride/size → scale-ratio mapping functions.

    All functions share ``f(r, scale_ratio) -> Tensor`` (same shape as ``r``),
    where ``r = |stride| / object_size`` (see ``select_candidates_in_gts``).
    """

    @classmethod
    def compute(cls, func_name: str, r: Tensor, scale_ratio):
        if func_name == 'static':
            return scale_ratio
        elif func_name == 'func_1':
            return cls.func_1(r, scale_ratio)
        elif func_name == 'func_2':
            return cls.func_2(r, scale_ratio)
        elif func_name == 'func_smooth_1':
            return cls.func_smooth_1(r, scale_ratio)
        elif func_name == 'func_gaussian_dip':
            return cls.func_gaussian_dip(r, scale_ratio[0], scale_ratio[1])
        elif func_name == 'func_exp_saturate':
            return cls.func_exp_saturate(r, scale_ratio)
        elif func_name == 'func_inverse_smooth':
            return cls.func_inverse_smooth(r, scale_ratio[0], scale_ratio[1],
                                           scale_ratio[2])
        elif func_name == 'func_scale_adaptive':
            base = scale_ratio[0] if isinstance(scale_ratio, (list, tuple)) else 1.0
            boost = scale_ratio[1] if isinstance(scale_ratio, (list, tuple)) else 0.5
            gamma = scale_ratio[2] if isinstance(scale_ratio, (list, tuple)) else 1.5
            return cls.func_scale_adaptive(r, base, boost, gamma)
        else:
            raise NotImplementedError(
                f'DScaleFunctions: unknown func_name={func_name!r}')

    @staticmethod
    def func_1(r, r_max):
        s = torch.zeros_like(r)
        s[(r >= 0.25) & (r < 1)] = r_max * (r[(r >= 0.25) & (r < 1)] - 0.25) / 0.75
        s[(r >= 1) & (r < 2)] = r_max
        s[(r >= 2) & (r < 2.5)] = r_max * (2.5 - r[(r >= 2) & (r < 2.5)])
        return s

    @staticmethod
    def func_2(r, r_max):
        s = torch.zeros_like(r)
        s[r < 1] = r_max * r[r < 1]
        s[r >= 1] = r_max
        return s

    @staticmethod
    def func_smooth_1(r, r_max, a=5.0, b=1.0):
        rise = 1 / (1 + torch.exp(-a * (r - 0.5)))
        fall = 1 - 0.25 / (1 + torch.exp(-b * (r - 2.0)))
        return r_max * rise * fall

    @staticmethod
    def func_gaussian_dip(r, r_max, r_ideal, sigma=0.1):
        return r_max * (1.0 - 0.5 * torch.exp(
            -torch.pow(r - r_ideal, 2) / (2 * sigma ** 2 + 1e-9)))

    @staticmethod
    def func_exp_saturate(r, r_max, a=1.0):
        return r_max * (1.0 - torch.exp(-a * r))

    @staticmethod
    def func_inverse_smooth(r, r_min, r_max, k):
        sig = 1 / (1 + torch.exp(-k * (r - 1)))
        return r_min + (r_max - r_min) * (1 - sig)

    @staticmethod
    def func_scale_adaptive(r, scale_base, scale_boost, gamma=2.0):
        return scale_base + scale_boost * torch.tanh(gamma * r)


# =============================================================================
# TaskAlignedAssignerDScaleDYAB
# =============================================================================

@TASK_UTILS.register_module()
class TaskAlignedAssignerDScaleDYAB(BaseAssigner):
    """Task-aligned assigner with dscale + dyab + area-refine soft label.

    Per-image port of ``TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea``.
    Designed for the RTMDet head (and any one-stage head that consumes
    ``AssignResult.max_overlaps`` as the soft label).

    Args:
        topk (int): Number of top candidate anchors per GT.  Defaults to 13.
        iou_calculator (ConfigType): Config of the IoU calculator used for all
            three roles (ownership / ranking / scoring).
            Defaults to ``dict(type='BboxOverlaps2D')``.
        eps (float): Numerical-stability epsilon.  Defaults to 1e-9.
        dscale_func (str): Key into :class:`DScaleFunctions`.  ``'static'``
            relaxes containment by ``min(l,t,r,b) > -(stride·scale_ratio)``
            (``scale_ratio=0`` → strict containment; ``scale_ratio=1`` → centre
            may lie one stride outside).
        scale_ratio (float | list | tuple): Scale-ratio argument for the dscale
            function (scalar for ``'static'``).  Defaults to 1.0.
        r_ref (float): Area-refine reference side length (ρ = 0.5 at
            area = r_ref²).  Defaults to 32.0.
        r_ref_type (str): Soft-label ceiling calibration mode.
            ``'pow'`` → ``ceil^ρ``;
            ``'add_1'`` → ``ceil + (1-ρ)·ceil·(1-ceil)``;
            anything else disables calibration.  Defaults to ``'pow'``.
        r_ref_use_adaptive (bool): If True, use the per-image median GT area as
            ``r_ref²`` instead of the fixed ``r_ref``.  Defaults to False.
        dyab_type (str): Key into ``DYAB_REGISTRY`` selecting the dynamic α/β
            strategy.  Defaults to ``'FixedAB'``.
        dyab_kwargs (dict): Keyword args for the Dyab subclass.  Defaults to
            ``dict(alpha=1.0, beta=6.0)``.
    """

    def __init__(self,
                 topk: int = 13,
                 iou_calculator: ConfigType = dict(type='BboxSiM2D',
                                                   mode='ciou'),
                 eps: float = 1e-9,
                 dscale_func: str = 'static',
                 scale_ratio=1.0,
                 r_ref: float = 32.0,
                 r_ref_type: str = 'pow',
                 r_ref_use_adaptive: bool = False,
                 dyab_type: str = 'FixedAB',
                 dyab_kwargs: Optional[dict] = None) -> None:
        assert topk >= 1
        self.topk = topk
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.eps = eps

        # ── dscale ─────────────────────────────────────────────────────────
        self.dscale_func = dscale_func
        self.scale_ratio = scale_ratio

        # ── area-refine soft label ─────────────────────────────────────────
        self.r_ref = r_ref
        self.r_ref_type = r_ref_type
        self.r_ref_use_adaptive = r_ref_use_adaptive

        # ── dyab ───────────────────────────────────────────────────────────
        dyab_kwargs = dyab_kwargs if dyab_kwargs is not None \
            else dict(alpha=1.0, beta=6.0)
        dyab_cls = DYAB_REGISTRY[dyab_type] if isinstance(dyab_type, str) \
            else dyab_type
        self.dyab = dyab_cls(**dyab_kwargs)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _gt_xyxy(gt_bboxes) -> Tensor:
        """Return GT boxes as an xyxy tensor (num_gt, 4)."""
        if isinstance(gt_bboxes, BaseBoxes):
            return gt_bboxes.tensor
        return gt_bboxes

    def _select_candidates_in_gts(self, centers: Tensor, strides: Tensor,
                                   gt_bboxes: Tensor) -> Tensor:
        """dscale candidate region.

        Args:
            centers (Tensor): anchor centres, (num_anchors, 2).
            strides (Tensor): per-anchor stride, (num_anchors,).
            gt_bboxes (Tensor): xyxy GT boxes, (num_gt, 4).

        Returns:
            Tensor[bool]: candidate mask, (num_gt, num_anchors).
        """
        num_gt = gt_bboxes.size(0)
        lt = gt_bboxes[:, :2]                       # (num_gt, 2)
        rb = gt_bboxes[:, 2:]                        # (num_gt, 2)
        # ltrb deltas: (num_gt, num_anchors, 4) = [l, t, r, b]
        lt_d = centers[None] - lt[:, None]           # (num_gt, A, 2)
        rb_d = rb[:, None] - centers[None]           # (num_gt, A, 2)
        deltas = torch.cat([lt_d, rb_d], dim=-1)     # (num_gt, A, 4)

        # ------------------------------------------------------------------
        # CRITICAL — sign of ``stride``.
        # In the ultralytics USAA implementation the assigner is fed a
        # *negative* stride tensor, so its ``deltas.amin > stride*scale_ratio``
        # and ``r = stride / (-w)`` expressions are evaluated with stride < 0.
        # mmdetection's ``priors[:, 2]`` stride is *positive*, so to reproduce
        # the original algorithm exactly we negate it here once and use the
        # signed (negative) stride everywhere below.
        # ------------------------------------------------------------------
        signed_stride = -strides[None, :]            # (1, A), < 0

        if self.dscale_func == 'static':
            # ultralytics: deltas.amin > stride*scale_ratio  with stride < 0
            #   -> deltas.amin > -(|stride|*scale_ratio)  (RELAX / expand)
            #   scale_ratio = 0 -> strict containment (centre inside the box)
            #   scale_ratio = 1 -> centre may lie up to one stride outside
            # Same convention as the validated two-stage port
            # ``DynamicSoftLabelAssignerDScaleDYAB``; admits more anchors for
            # tiny objects, which is essential on AITOD.
            thresh = signed_stride * self.scale_ratio             # (1, A), <= 0
            return deltas.amin(dim=-1) > thresh                    # (num_gt, A)

        # ---- size-dependent containment threshold ----
        # Faithful to ultralytics (mla_usaa.select_candidates_in_gts), using the
        # signed (negative) stride: r = stride / (-w) is positive, and
        # thresh = stride * f(r) is <= 0 (expansion).
        w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=self.eps)  # (num_gt,)
        h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=self.eps)  # (num_gt,)
        r_w = signed_stride / (-w[:, None] + self.eps)             # (num_gt, A), > 0
        r_h = signed_stride / (-h[:, None] + self.eps)
        thresh_w = signed_stride * DScaleFunctions.compute(
            self.dscale_func, r_w, self.scale_ratio)
        thresh_h = signed_stride * DScaleFunctions.compute(
            self.dscale_func, r_h, self.scale_ratio)
        mask_in = (deltas[..., 0] > thresh_w) & (deltas[..., 2] > thresh_w) \
            & (deltas[..., 1] > thresh_h) & (deltas[..., 3] > thresh_h)
        return mask_in

    def _compute_alpha_beta(self, gt_bboxes: Tensor,
                            num_gt: int) -> Tuple[Tensor, Tensor]:
        """Compute per-GT (α, β) and reshape to broadcast over (num_gt, A)."""
        alpha, beta = self.dyab.compute(
            None, gt_bboxes, num_gt, num_gt, device=gt_bboxes.device)

        def _reshape(v):
            if isinstance(v, (int, float)):
                return float(v)
            # per-GT 1-D tensor → (num_gt, 1) to broadcast over anchors
            return v.reshape(num_gt, 1)

        return _reshape(alpha), _reshape(beta)

    def _select_topk_candidates(self, metrics: Tensor) -> Tensor:
        """Top-k anchors per GT, with the duplicate-index filtering used by
        ultralytics' ``select_topk_candidates``.

        Args:
            metrics (Tensor): alignment metric, (num_gt, num_anchors).

        Returns:
            Tensor: top-k mask, (num_gt, num_anchors).
        """
        num_gt, na = metrics.shape
        topk = min(self.topk, na)
        topk_metrics, topk_idxs = torch.topk(
            metrics, topk, dim=-1, largest=True)        # (num_gt, topk)

        count = torch.zeros_like(metrics, dtype=torch.int8)
        ones = torch.ones_like(topk_idxs[:, :1], dtype=torch.int8)
        for k in range(topk):
            count.scatter_add_(-1, topk_idxs[:, k:k + 1], ones)
        # drop anchors that were selected for the same GT more than once
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    @staticmethod
    def _select_highest_overlaps(mask_pos: Tensor, overlaps: Tensor,
                                 num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Resolve anchors matched to multiple GTs by highest IoU.

        Args & returns mirror ultralytics ``select_highest_overlaps`` but in
        the per-image ``(num_gt, num_anchors)`` layout.
        """
        fg_mask = mask_pos.sum(dim=0)                    # (num_anchors,)
        if fg_mask.max() > 1:
            mask_multi = (fg_mask.unsqueeze(0) > 1).expand(num_gt, -1)
            max_overlaps_idx = overlaps.argmax(dim=0)    # (num_anchors,)
            is_max = torch.zeros_like(mask_pos)
            is_max.scatter_(0, max_overlaps_idx.unsqueeze(0), 1)
            mask_pos = torch.where(mask_multi, is_max, mask_pos).float()
            fg_mask = mask_pos.sum(dim=0)
        target_gt_idx = mask_pos.argmax(dim=0)           # (num_anchors,)
        return target_gt_idx, fg_mask, mask_pos

    def _compute_rho(self, gt_bboxes: Tensor) -> Tensor:
        """Per-GT reliability ratio ρ = area / (area + r_ref²), shape (num_gt,)."""
        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
        r_sq = gt_w * gt_h
        if self.r_ref_use_adaptive and r_sq.numel() > 0:
            r_ref_sq = r_sq.median().detach()
        else:
            r_ref_sq = self.r_ref ** 2
        return r_sq / (r_sq + r_ref_sq)

    def _calibrate_ceiling(self, pos_overlaps: Tensor,
                           rho: Tensor) -> Tensor:
        """Area-refine calibration of the per-GT soft-label IoU ceiling.

        Args:
            pos_overlaps (Tensor): per-GT max overlap, (1, num_gt).
            rho (Tensor): per-GT reliability ratio, (num_gt,).
        """
        rho = rho[None, :]                               # (1, num_gt)
        if self.r_ref_type == 'pow':
            return pos_overlaps.pow(rho)
        elif self.r_ref_type == 'add_1':
            return pos_overlaps + (1.0 - rho) * pos_overlaps * (1.0 - pos_overlaps)
        return pos_overlaps

    # ── main assign ──────────────────────────────────────────────────────────

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign GTs to anchors and return an :class:`AssignResult`.

        ``max_overlaps`` carries the per-anchor ``norm_align_metric`` (the soft
        label consumed by the RTMDet head).
        """
        priors = pred_instances.priors                   # (A, 4): cx,cy,sw,sh
        decoded_bboxes = pred_instances.bboxes           # (A, 4)
        pred_scores = pred_instances.scores              # (A, C) logits
        gt_bboxes_raw = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes = self._gt_xyxy(gt_bboxes_raw)

        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1, dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        centers = priors[:, :2]
        strides = priors[:, 2]
        # ultralytics feeds the assigner sigmoid probabilities, not logits
        scores = pred_scores.sigmoid()

        # ── dscale: candidate region, (num_gt, A) ──────────────────────────
        mask_in_gts = self._select_candidates_in_gts(centers, strides,
                                                      gt_bboxes)

        # ── box metrics (alignment metric + dyab), (num_gt, A) ──────────────
        # cls score of each anchor for each GT's class
        bbox_scores = scores[:, gt_labels].t()           # (num_gt, A)
        # clamp to >=0: CIoU/DIoU/GIoU can be negative, but the alignment
        # metric (overlaps**beta) and the soft-label ceiling need IoU in [0, 1]
        # (mirrors ultralytics' ``iou_calculation(...).clamp_(0)``).
        overlaps = self.iou_calculator(
            decoded_bboxes, gt_bboxes).clamp(min=0).t()   # (num_gt, A)

        # restrict metrics to the candidate region (mask_gt is all-true here)
        cand = mask_in_gts.to(bbox_scores.dtype)
        bbox_scores = bbox_scores * cand
        overlaps = overlaps * cand

        alpha, beta = self._compute_alpha_beta(gt_bboxes, num_gt)
        align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)

        # ── top-k candidates + merge masks ──────────────────────────────────
        mask_topk = self._select_topk_candidates(align_metric)
        mask_pos = mask_topk * mask_in_gts.to(mask_topk.dtype)   # (num_gt, A)

        # ── resolve multi-GT anchors by highest IoU ─────────────────────────
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, num_gt)
        fg_inds = fg_mask.bool()

        # ── soft label (+ area-refine), per anchor ──────────────────────────
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # (num_gt,1)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # (num_gt,1)
        # reshape to (1, num_gt) for the ceiling-calibration helper
        pos_overlaps_row = pos_overlaps.reshape(1, num_gt)
        rho = self._compute_rho(gt_bboxes)
        pos_overlaps_cal = self._calibrate_ceiling(pos_overlaps_row, rho)
        pos_overlaps_cal = pos_overlaps_cal.reshape(num_gt, 1)

        norm_align_metric = (align_metric * pos_overlaps_cal /
                             (pos_align_metrics + self.eps)).amax(dim=0)  # (A,)

        # ── build AssignResult ──────────────────────────────────────────────
        max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
        assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                  -1, dtype=torch.long)
        if fg_inds.any():
            assigned_gt_inds[fg_inds] = target_gt_idx[fg_inds] + 1
            assigned_labels[fg_inds] = gt_labels[target_gt_idx[fg_inds]].long()
            max_overlaps[fg_inds] = norm_align_metric[fg_inds]

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
