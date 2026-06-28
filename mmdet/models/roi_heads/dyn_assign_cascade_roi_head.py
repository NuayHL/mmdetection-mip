# Copyright (c) OpenMMLab. All rights reserved.
"""Cascade RoI head with prediction-aware (dynamic soft label) assigner support.

Extends :class:`CascadeRoIHead` so that each cascade stage can use a
prediction-aware assigner (:class:`DynamicSoftLabelAssigner` and its
subclasses) while keeping the cascade refinement mechanism intact.

The standard :class:`CascadeRoIHead` only supports "static" assigners
(:class:`MaxIoUAssigner`) that require nothing beyond the proposal boxes.
Prediction-aware assigners need decoded bboxes and cls scores, which forces
a no-grad pre-assignment forward before each stage's main forward.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.losses.accuracy import accuracy
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import unpack_gt_instances
from .cascade_roi_head import CascadeRoIHead


@MODELS.register_module()
class DynAssignCascadeRoIHead(CascadeRoIHead):
    """Cascade RoI head supporting prediction-aware label assigners per stage.

    Compared with :class:`CascadeRoIHead` this head inserts an extra no-grad
    "pre-assignment" bbox forward over **all** proposals at the beginning of
    every cascade stage so that prediction-aware assigners (e.g.
    :class:`DynamicSoftLabelAssigner`, :class:`DynamicSoftLabelAssignerAreaRefine`,
    :class:`DynamicSoftLabelAssignerDScaleDYAB`) can read decoded boxes and
    cls scores from ``pred_instances``.

    Each cascade stage follows the same two-pass pattern as
    :class:`DynAssignRoIHead`::

        Forward 1 (no_grad) on all current proposals
            -> cls_score_all, bbox_pred_all
        build pred_instances per image with {priors, bboxes, scores}
        assigner.assign(pred_instances, gt) -> AssignResult
        sampler.sample(...)                 -> SamplingResult
        Forward 2 (with grad) on sampled priors -> loss

    Between stages, :class:`CascadeRoIHead`'s standard bbox refinement is
    applied, so later stages operate on progressively improved proposals.

    Args:
        cls_score_activation (str): How to turn raw cls logits into scores
            for the assigner. One of ``'softmax'``, ``'sigmoid'``,
            ``'identity'``. Defaults to ``'softmax'``.
        prior_format (str): Spatial layout of ``pred_instances.priors``.
            ``'xyxy'`` passes proposals as-is; ``'point'`` converts to
            ``(cx, cy, stride, stride)``.  Defaults to ``'xyxy'``.
        use_iou_soft_target (bool): If True, ``AssignResult.max_overlaps``
            is forwarded to ``loss_cls`` as a soft IoU target (requires
            ``QualityFocalLoss``). Defaults to False.
    """

    def __init__(self,
                 cls_score_activation: str = 'softmax',
                 prior_format: str = 'xyxy',
                 use_iou_soft_target: bool = False,
                 min_proposal_size: float = 1.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert cls_score_activation in ('softmax', 'sigmoid', 'identity'), (
            f"cls_score_activation must be one of 'softmax', 'sigmoid', "
            f"'identity', got {cls_score_activation!r}")
        assert prior_format in ('xyxy', 'point'), (
            f"prior_format must be 'xyxy' or 'point', got {prior_format!r}")
        self.cls_score_activation = cls_score_activation
        self.prior_format = prior_format
        self.use_iou_soft_target = use_iou_soft_target
        # Floor every proposal's width/height (px) before it is used for
        # assignment / target encoding. ``refine_bboxes`` border-clamps
        # out-of-image predictions, which can collapse a box to ZERO width or
        # height; encoding such a proposal gives ``dw = log(gw/0) = inf``
        # (``bbox2delta`` has no guard) → inf regression target → nan grads →
        # dead net. This is the actual NaN root cause on cascade/detectors
        # (single-stage Faster R-CNN has no refine step, so it never hits it).
        # Kept LOCAL to this head (rather than patching the global bbox coder).
        # A sub-pixel proposal is meaningless, so this never touches a real
        # detection. Set to 0 to disable.
        self.min_proposal_size = min_proposal_size

    # ------------------------------------------------------------------
    #  Helpers  (mirror DynAssignRoIHead)
    # ------------------------------------------------------------------

    def _activate_cls_score(self, cls_score: Tensor) -> Tensor:
        """Map raw cls logits to per-class scores."""
        num_classes = self.bbox_head[0].num_classes
        if self.cls_score_activation == 'softmax':
            scores = F.softmax(cls_score, dim=-1)
        elif self.cls_score_activation == 'sigmoid':
            scores = cls_score.sigmoid()
        else:
            scores = cls_score
        if scores.shape[-1] == num_classes + 1:
            scores = scores[..., :num_classes]
        return scores

    @staticmethod
    def _xyxy_to_point_priors(xyxy: Tensor) -> Tensor:
        """Convert XYXY boxes to ``(cx, cy, stride, stride)`` point format."""
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1.0)
        h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1.0)
        stride = torch.minimum(w, h)
        return torch.stack([cx, cy, stride, stride], dim=-1)

    def _decode_for_assignment(self, priors: Tensor, bbox_pred: Tensor,
                               scores: Tensor,
                               bbox_head) -> Tensor:
        """Decode regression output into absolute boxes for the assigner."""
        if bbox_head.reg_class_agnostic:
            return bbox_head.bbox_coder.decode(priors, bbox_pred)

        n = bbox_pred.size(0)
        c = bbox_head.num_classes
        bbox_pred = bbox_pred.view(n, c, -1)
        pred_cls = scores.argmax(dim=-1)
        chosen = bbox_pred[torch.arange(n, device=bbox_pred.device), pred_cls]
        return bbox_head.bbox_coder.decode(priors, chosen)

    @torch.no_grad()
    def _bbox_forward_for_assignment(
            self, stage: int, x: Tuple[Tensor],
            rois: Tensor) -> Tuple[Tensor, Tensor]:
        """No-grad bbox forward for assignment on ALL proposals of a stage."""
        bbox_results = self._bbox_forward(stage, x, rois)
        return bbox_results['cls_score'], bbox_results['bbox_pred']

    @staticmethod
    def _is_prediction_aware(assigner) -> bool:
        """Return True if the assigner needs model predictions."""
        from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner \
            import DynamicSoftLabelAssigner
        return isinstance(assigner, DynamicSoftLabelAssigner)

    def _clamp_proposals_min_size(self, results_list: InstanceList) -> None:
        """Floor proposal w/h in place so target encoding can't produce inf.

        Guards against degenerate (zero/sub-pixel) proposals from
        ``refine_bboxes`` border-clamping; see ``min_proposal_size``. Operates
        on the underlying tensor (works for raw Tensor and BaseBoxes alike), so
        the box type is preserved. Clamps both ``priors`` and ``bboxes`` if
        present, since either may carry the proposal boxes at a given stage.
        """
        ms = self.min_proposal_size
        if ms <= 0:
            return
        for res in results_list:
            for key in ('priors', 'bboxes'):
                boxes = res.get(key, None)
                if boxes is None:
                    continue
                t = get_box_tensor(boxes)
                if t.numel() == 0:
                    continue
                t[:, 2] = torch.maximum(t[:, 2], t[:, 0] + ms)
                t[:, 3] = torch.maximum(t[:, 3], t[:, 1] + ms)

    # ------------------------------------------------------------------
    #  IoU-target loss  (mirror DynAssignRoIHead._loss_with_iou_target)
    # ------------------------------------------------------------------

    @staticmethod
    def _loss_with_iou_target(bbox_head, cls_score: Tensor,
                              bbox_pred: Tensor, rois: Tensor,
                              labels: Tensor, label_weights: Tensor,
                              bbox_targets: Tensor, bbox_weights: Tensor,
                              iou_target: Tensor) -> dict:
        """Compute bbox losses with soft IoU targets for QFL.

        The classification branch receives a ``(labels, iou_target)`` tuple
        (the contract used by :class:`QualityFocalLoss`).  The regression
        branch is identical to the standard path.
        """
        losses: dict = dict()

        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(
                torch.sum(label_weights > 0).float().item(), 1.)
            loss_cls_ = bbox_head.loss_cls(
                cls_score, (labels, iou_target),
                label_weights,
                avg_factor=avg_factor)
            if isinstance(loss_cls_, dict):
                losses.update(loss_cls_)
            else:
                losses['loss_cls'] = loss_cls_
            losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = bbox_head.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if bbox_head.reg_decoded_bbox:
                    bbox_pred = bbox_head.bbox_coder.decode(
                        rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if bbox_head.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), bbox_head.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = bbox_head.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0))
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses

    # ------------------------------------------------------------------
    #  Core training logic
    # ------------------------------------------------------------------

    def bbox_loss(self,
                  stage: int,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  pos_iou_targets_list: Optional[List[Tensor]] = None) -> dict:
        """Forward sampled priors and compute bbox losses for one stage.

        When ``use_iou_soft_target`` is True and ``pos_iou_targets_list`` is
        given, the classification loss receives soft IoU targets (QFL path).
        Otherwise falls back to the standard ``loss_and_target`` path.
        """
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_results.update(rois=rois)

        if not self.use_iou_soft_target or pos_iou_targets_list is None:
            bbox_loss_and_target = bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score'],
                bbox_pred=bbox_results['bbox_pred'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg[stage])
            bbox_results.update(bbox_loss_and_target)
            return bbox_results

        # ---- IoU soft target path ----
        labels, label_weights, bbox_targets, bbox_weights = \
            bbox_head.get_targets(
                sampling_results, self.train_cfg[stage], concat=True)

        # Populate iou_target matching the [pos..., neg...] layout of
        # _get_targets_single / get_targets concatenation.
        iou_target = bbox_results['cls_score'].new_zeros(labels.shape[0])
        offset = 0
        for samp, pos_ious in zip(sampling_results, pos_iou_targets_list):
            num_pos = samp.pos_inds.numel()
            num_neg = samp.neg_inds.numel()
            if num_pos > 0:
                iou_target[offset:offset + num_pos] = pos_ious
            offset += num_pos + num_neg
        assert offset == labels.shape[0], (
            f'iou_target layout mismatch: filled {offset} of {labels.shape[0]}')

        losses = self._loss_with_iou_target(
            bbox_head,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            labels, label_weights, bbox_targets, bbox_weights,
            iou_target)
        bbox_results.update(loss_bbox=losses)
        # CascadeRoIHead.loss() expects bbox_targets for refine_bboxes()
        bbox_results['bbox_targets'] = (labels, label_weights,
                                        bbox_targets, bbox_weights)
        return bbox_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Cascade loss with optional pre-assignment forward per stage.

        For stages whose assigner is prediction-aware, we run a no-grad
        forward on all current proposals, decode the predictions, and feed
        the resulting :class:`InstanceData` to the assigner.  The rest
        (sampling, loss, refinement) follows the standard cascade flow.
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        num_imgs = len(batch_data_samples)
        losses = dict()
        results_list = rpn_results_list

        for stage in range(self.num_stages):
            self.current_stage = stage
            stage_loss_weight = self.stage_loss_weights[stage]

            # Floor degenerate (zero/sub-pixel) proposals before they are used
            # for the pre-assignment forward, assignment and target encoding —
            # otherwise a border-clamped refined box gives an inf regression
            # target (see _clamp_proposals_min_size). Applied to every stage's
            # proposals (RPN at stage 0, refined boxes at later stages).
            self._clamp_proposals_min_size(results_list)

            bbox_assigner = self.bbox_assigner[stage]
            bbox_sampler = self.bbox_sampler[stage]
            prediction_aware = self._is_prediction_aware(bbox_assigner)

            sampling_results: List[SamplingResult] = []
            pos_iou_targets_list: List[Tensor] = []

            # ---  Pre-assignment forward (no_grad) when needed  ---
            if prediction_aware:
                # Grab current proposals (already refined for stages > 0)
                proposals = []
                for r in results_list:
                    # Stage 1: rpn_results_list may have 'bboxes' or 'priors'
                    # Later stages: refine_bboxes returns InstanceData with
                    # 'bboxes' and 'priors' set in the next iteration.
                    p = r.get('priors', None)
                    if p is None:
                        p = r.bboxes
                    proposals.append(p)
                num_props_per_img = [p.shape[0] for p in proposals]

                rois_all = bbox2roi(proposals)
                cls_score_all, bbox_pred_all = \
                    self._bbox_forward_for_assignment(stage, x, rois_all)
                cls_score_per_img = cls_score_all.split(
                    num_props_per_img, 0)
                bbox_pred_per_img = bbox_pred_all.split(
                    num_props_per_img, 0)

            # ---  Per-image assignment + sampling  ---
            for i in range(num_imgs):
                results = results_list[i]
                # Standard cascade pop: 'bboxes' → 'priors'
                results.priors = results.pop('bboxes')

                if prediction_aware:
                    priors_i = results.priors
                    cls_score_i = cls_score_per_img[i]
                    bbox_pred_i = bbox_pred_per_img[i]

                    scores_i = self._activate_cls_score(cls_score_i)
                    decoded_i = self._decode_for_assignment(
                        priors_i, bbox_pred_i, scores_i,
                        self.bbox_head[stage])
                    decoded_i = get_box_tensor(decoded_i)

                    assign_priors = (self._xyxy_to_point_priors(priors_i)
                                     if self.prior_format == 'point'
                                     else priors_i)
                    pred_for_assign = InstanceData()
                    pred_for_assign.priors = assign_priors
                    pred_for_assign.bboxes = decoded_i
                    pred_for_assign.scores = scores_i

                    assign_result = bbox_assigner.assign(
                        pred_for_assign, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])
                else:
                    assign_result = bbox_assigner.assign(
                        results, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])

                sampling_result = bbox_sampler.sample(
                    assign_result, results, batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                if prediction_aware and self.use_iou_soft_target:
                    pos_iou_targets_list.append(
                        assign_result.max_overlaps[
                            sampling_result.pos_inds])

            # ---  Bbox loss  ---
            bbox_results = self.bbox_loss(
                stage, x, sampling_results,
                pos_iou_targets_list if prediction_aware else None)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # ---  Refine bboxes for the next stage  ---
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results, bbox_results, batch_img_metas)
                    if results_list is None:
                        break

        return losses
