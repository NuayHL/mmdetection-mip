# Copyright (c) OpenMMLab. All rights reserved.
"""
MaxIoUPredSoftLabelAssigner — the most conservative *task-aligned* softening of
:class:`MaxIoUAssigner`.

Motivation
----------
RFLA / NWD-RKA succeed because the *label* a sample receives reflects its
**actual quality** (a smooth distance metric), not a hard 0/1. Those methods
live entirely in the RPN, though — they never touch the RCNN head. The two
prediction-aware RCNN assigners tried so far either underperform plain MaxIoU
when combined with RFLA (``TaskAlignedAssignerUSAA``,
``DynamicSoftLabelAssignerAreaRefine``) because they change **both** the
candidate *selection* and the *label*; that double change is what destabilises
the combination.

This assigner makes the **minimum** change that still imports RFLA's idea into
the RCNN head:

  * The hard pos/neg **assignment is byte-for-byte** :class:`MaxIoUAssigner`
    (IoU thresholding on the *proposals* + optional low-quality matching).
    This is the part proven to combine with an RFLA / NWD-RKA RPN, so it is
    left untouched → minimal debuff.

  * Only the *positive* classification **target** is softened. Instead of the
    hard ``1`` (or the static proposal-IoU of :class:`MaxSoftIoUAssigner`), a
    positive receives::

        soft = max( calibrate( IoU(decoded_pred_bbox, matched_gt) ),  pos_iou_thr )

    where

      - ``decoded_pred_bbox`` is the box the RCNN bbox head **predicts** for
        that proposal (read from ``pred_instances.bboxes``; requires the
        two-pass :class:`DynAssignRoIHead`). This is what makes the label
        *task-aligned* — it measures how well the head is actually localising,
        not just how good the RPN proposal was.

      - ``calibrate`` is the per-GT area-refine lift (identical to
        :class:`MaxSoftIoUAssigner` / ``TaskAlignedAssignerUSAA``) that raises
        the soft-label ceiling for small objects::

            rho = area / (area + r_ref**2)
              pow:    cal = iou ** rho
              add_1:  cal = iou + lambda * (1 - rho) * iou * (1 - iou)

      - the ``max(·, pos_iou_thr)`` **floor** guarantees every positive keeps a
        target of at least ``pos_iou_thr``. This is the "no big debuff" safety
        net: even when the head's early prediction is poor (low decoded IoU),
        the positive is never driven below the hard-assignment threshold, so
        the positive gradient cannot collapse (the failure mode of the additive
        DSL port, where tiny-object positives got ~0 soft labels).

Contrast with :class:`MaxSoftIoUAssigner`
-----------------------------------------
``MaxSoftIoUAssigner`` calibrates the **proposal** IoU (prediction-independent,
no floor). This assigner calibrates the **prediction** IoU (task-aligned) and
floors it. Set ``score_mode='iou'`` for the standard recipe; switch it to
``'nwd'`` / ``'kl'`` to make the RCNN soft-target *currency* match an
NWD-RKA / RFLA RPN.

Wiring
------
Must be paired with ``DynAssignRoIHead`` so the decoded prediction is
available, and with a ``QualityFocalLoss`` cls head so the soft target is
consumed::

    roi_head=dict(
        type='DynAssignRoIHead',
        cls_score_activation='sigmoid',
        prior_format='xyxy',          # super().assign reads priors as xyxy boxes
        use_iou_soft_target=True,
        bbox_head=dict(
            loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True,
                          beta=0.0, loss_weight=1.0, custom_cls_channels=True)))
    train_cfg=dict(rcnn=dict(
        assigner=dict(type='MaxIoUPredSoftLabelAssigner',
                      pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                      match_low_quality=False,
                      r_ref=32.0, calibrate_mode='add_1', score_mode='iou')))

If ``pred_instances.bboxes`` is absent (e.g. used under a plain
``StandardRoIHead``), the assigner gracefully falls back to the proposal IoU
(behaving like a floored :class:`MaxSoftIoUAssigner`), so it stays a safe
drop-in.
"""

from typing import Optional, Union

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner


@TASK_UTILS.register_module()
class MaxIoUPredSoftLabelAssigner(MaxIoUAssigner):
    """MaxIoU hard assignment + a *prediction-aware*, floored soft label.

    The pos/neg selection is exactly :class:`MaxIoUAssigner`. Only the
    per-positive ``AssignResult.max_overlaps`` (the soft classification target
    consumed by ``QualityFocalLoss`` via ``DynAssignRoIHead``) is replaced by::

        max( calibrate( metric(decoded_pred_bbox, matched_gt) ),  pos_iou_thr )

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes (also the soft
            label floor).
        neg_iou_thr (float | tuple): IoU threshold for negative bboxes.
        r_ref (float): Reference object side length (px). ``rho = 0.5`` at
            ``area = r_ref**2``. Defaults to 32.0.
        calibrate_mode (str): ``'pow'`` → ``iou ** rho``,
            ``'add_1'`` → ``iou + lambda * (1 - rho) * iou * (1 - iou)``,
            ``'none'`` → no calibration. Defaults to ``'add_1'``.
        lambda_refine (float): Strength of the ``add_1`` lift. Defaults to 1.0.
        score_mode (str): Metric used for the soft label between the decoded
            prediction and its matched GT. Any :class:`BboxDistanceMetric`
            mode (``'iou'`` / ``'giou'`` / ``'nwd'`` / ``'kl'`` / ``'wd'`` /
            ``'dotd'`` …). Defaults to ``'iou'``.
        score_iou_calculator (dict): Calculator producing the ``score_mode``
            metric. Defaults to ``dict(type='BboxDistanceMetric')``.
        floor_to_pos_thr (bool): Apply the ``max(·, pos_iou_thr)`` floor.
            Defaults to True.
        soft_label (bool): If False, leaves ``max_overlaps`` as the plain
            proposal IoU (pure MaxIoU ablation). Defaults to True.
        **kwargs: Forwarded to :class:`MaxIoUAssigner` (``min_pos_iou``,
            ``match_low_quality``, ``gpu_assign_thr``, ``iou_calculator`` …).
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: Union[float, tuple],
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'add_1',
                 lambda_refine: float = 1.0,
                 score_mode: str = 'iou',
                 score_iou_calculator: dict = dict(type='BboxDistanceMetric'),
                 floor_to_pos_thr: bool = True,
                 soft_label: bool = True,
                 **kwargs):
        super().__init__(
            pos_iou_thr=pos_iou_thr, neg_iou_thr=neg_iou_thr, **kwargs)
        assert calibrate_mode in ('pow', 'add_1', 'none'), \
            f'calibrate_mode must be pow/add_1/none, got {calibrate_mode!r}'
        self.r_ref_sq = r_ref ** 2
        self.calibrate_mode = calibrate_mode
        self.lambda_refine = lambda_refine
        self.score_mode = score_mode
        self.score_iou_calculator = TASK_UTILS.build(score_iou_calculator)
        self.floor_to_pos_thr = floor_to_pos_thr
        self.soft_label = soft_label

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _areas(bboxes) -> torch.Tensor:
        if isinstance(bboxes, BaseBoxes):
            return bboxes.areas
        w = (bboxes[:, 2] - bboxes[:, 0]).clamp(min=1.0)
        h = (bboxes[:, 3] - bboxes[:, 1]).clamp(min=1.0)
        return w * h

    def _calibrate(self, iou: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Per-positive area-refine lift; ``iou`` and ``rho`` are aligned."""
        if self.calibrate_mode == 'pow':
            return iou.pow(rho)
        if self.calibrate_mode == 'add_1':
            return iou + self.lambda_refine * (1.0 - rho) * iou * (1.0 - iou)
        return iou

    # ── main assign ───────────────────────────────────────────────────────

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Standard MaxIoU assignment, then a prediction-aware soft label."""
        assign_result = super().assign(pred_instances, gt_instances,
                                       gt_instances_ignore, **kwargs)
        if not self.soft_label:
            return assign_result

        gt_bboxes = gt_instances.bboxes
        num_gt = gt_bboxes.size(0)
        gt_inds = assign_result.gt_inds
        max_overlaps = assign_result.max_overlaps
        pos_mask = gt_inds > 0
        if num_gt == 0 or not pos_mask.any():
            return assign_result

        # 0-based GT index for each positive proposal.
        pos_gt_idx = gt_inds[pos_mask] - 1

        # Source IoU for the soft label.
        #   * task-aligned: IoU(decoded prediction, matched GT) — needs the
        #     two-pass DynAssignRoIHead to populate pred_instances.bboxes.
        #   * fallback: proposal IoU already in max_overlaps (drop-in safety).
        decoded = getattr(pred_instances, 'bboxes', None)
        if decoded is not None:
            decoded = get_box_tensor(decoded)
            pos_decoded = decoded[pos_mask]
            # (num_pos, num_gt) metric, then gather the matched-GT column.
            metric = self.score_iou_calculator(
                pos_decoded, gt_bboxes, mode=self.score_mode)
            rows = torch.arange(pos_decoded.size(0), device=metric.device)
            iou_pos = metric[rows, pos_gt_idx]
        else:
            iou_pos = max_overlaps[pos_mask]
        iou_pos = iou_pos.clamp(min=0.0, max=1.0)

        # Per-GT reliability ratio rho = area / (area + r_ref**2).
        rho = self._areas(gt_bboxes) / (self._areas(gt_bboxes) + self.r_ref_sq)
        rho_pos = rho.to(iou_pos.device)[pos_gt_idx]

        soft = self._calibrate(iou_pos, rho_pos).clamp(min=0.0, max=1.0)
        if self.floor_to_pos_thr:
            soft = torch.clamp(soft, min=float(self.pos_iou_thr))

        max_overlaps[pos_mask] = soft
        assign_result.max_overlaps = max_overlaps
        return assign_result
