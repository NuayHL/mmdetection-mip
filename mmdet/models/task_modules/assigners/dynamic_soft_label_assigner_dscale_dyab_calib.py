# Copyright (c) OpenMMLab. All rights reserved.
"""
DynamicSoftLabelAssignerDScaleDYABCalib — DSL-DScale-DYAB whose *delivered
supervision signal* (the soft IoU target the loss trains on) is area-refine
calibrated, not just the internal matching cost.

Why this exists
---------------
In :class:`DynamicSoftLabelAssignerDScaleDYAB` the area-refine calibration
``soft_label_ious = _calibrate_iou(pairwise_ious, rho)`` feeds ONLY the matching
cost matrix. The value finally written to ``AssignResult.max_overlaps`` — i.e.
the soft classification target the RoI head forwards to ``QualityFocalLoss`` via
``use_iou_soft_target`` — is the RAW matched IoU gathered by
``dynamic_k_matching`` (it is handed ``pairwise_ious``, not ``soft_label_ious``).
So the head that "works on the baseline" never actually trains on a *calibrated*
target; the calibration only steers which proposals become positive.

This subclass closes that gap with the smallest possible change:
  * Matching is byte-for-byte the parent's (we call ``super().assign``), so the
    behaviour that already gives large gains on Faster / Cascade / Detectors is
    preserved exactly.
  * Only the positive ``max_overlaps`` is calibrated, with the SAME
    ``_calibrate_iou`` / ``_compute_rho`` the parent already uses for the cost.
    After ``super().assign`` the raw matched IoU is already sitting in
    ``max_overlaps[pos]``, so we calibrate it in place — no re-decoding, no
    re-matching, no change to the positive/negative selection.

Result: the soft label delivered to the loss is now area-refine calibrated,
making the "calibrated soft label" story honest on the two-stage baseline (the
spirit of USAA's calibrated supervision, realised on the additive-cost DSL head
that actually trains well on plain Faster R-CNN).
"""

from typing import Optional

from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .dynamic_soft_label_assigner_dscale_dyab import \
    DynamicSoftLabelAssignerDScaleDYAB


@TASK_UTILS.register_module()
class DynamicSoftLabelAssignerDScaleDYABCalib(DynamicSoftLabelAssignerDScaleDYAB):
    """DSL-DScale-DYAB that delivers a *calibrated* soft IoU target.

    Identical matching to :class:`DynamicSoftLabelAssignerDScaleDYAB`. The only
    difference: the per-positive soft label written to
    ``AssignResult.max_overlaps`` is area-refine calibrated (the very
    calibration the parent applies to the matching cost) instead of the raw
    matched IoU.

    Args:
        calibrate_target (bool): If True (default) calibrate the delivered soft
            label. If False, behaves exactly like the parent (raw-IoU target) —
            the clean A/B ablation isolating "calibrated supervision signal".
        **kwargs: Forwarded to :class:`DynamicSoftLabelAssignerDScaleDYAB`
            (``calibrate_mode``, ``r_ref``, ``dyab_type`` …). The supervision
            calibration reuses the inherited ``calibrate_mode`` / ``r_ref``, so
            the matching cost and the delivered label share one calibration.
    """

    def __init__(self, calibrate_target: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calibrate_target = calibrate_target

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Run the parent matching, then calibrate the positive soft labels."""
        assign_result = super().assign(pred_instances, gt_instances,
                                       gt_instances_ignore, **kwargs)
        if not self.calibrate_target:
            return assign_result

        gt_bboxes = gt_instances.bboxes
        num_gt = gt_bboxes.size(0)
        gt_inds = assign_result.gt_inds
        pos_mask = gt_inds > 0
        if num_gt == 0 or not pos_mask.any():
            return assign_result

        # The raw matched IoU is already in max_overlaps[pos] (written by the
        # parent's dynamic_k_matching). Calibrate it with the same area-refine
        # calibration the parent uses for the cost matrix.
        pos_gt_idx = gt_inds[pos_mask] - 1                       # (num_pos,)
        rho = self._compute_rho(gt_bboxes)                       # (num_gt,)
        rho_pos = rho.to(gt_inds.device)[pos_gt_idx]             # (num_pos,)

        raw_iou = assign_result.max_overlaps[pos_mask].clamp(0.0, 1.0)
        # _calibrate_iou is elementwise after rho[None, :]; feed (1, num_pos).
        cal = self._calibrate_iou(raw_iou[None, :], rho_pos)[0]
        assign_result.max_overlaps[pos_mask] = cal.clamp(0.0, 1.0)
        return assign_result