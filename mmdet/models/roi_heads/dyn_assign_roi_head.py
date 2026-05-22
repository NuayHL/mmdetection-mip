# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.losses.accuracy import accuracy
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class DynAssignRoIHead(StandardRoIHead):
    """RoI head supporting prediction-aware (dynamic) label assigners.

    Compared with :class:`StandardRoIHead` this head inserts an extra
    "pre-assignment" bbox forward pass over **all** RPN proposals so that
    prediction-aware assigners (SimOTA / DynamicSoftLabel / TaskAligned)
    can read decoded boxes and cls scores from ``pred_instances``.

    Training flow per image::

        Forward 1 (no_grad) on all RPN proposals
            -> cls_score_all, bbox_pred_all
        build pred_instances = {priors, bboxes=decoded, scores=activated}
        assigner.assign(pred_instances, gt) -> AssignResult
        sampler.sample(...)                  -> SamplingResult
            (supports add_gt_as_proposals)
        Forward 2 (with grad) on sampling_result.priors
            -> cls_score, bbox_pred
        loss(cls_score, bbox_pred, targets [+ iou_target])

    Args:
        cls_score_activation (str): How to turn the raw cls logits into the
            ``scores`` field consumed by the assigner. One of:

              - ``'softmax'`` (default): softmax then drop the bg channel.
                Use with the default ``CrossEntropyLoss(use_sigmoid=False)``.
              - ``'sigmoid'``: elementwise sigmoid. Use with sigmoid losses
                such as ``QualityFocalLoss``.
              - ``'identity'``: pass logits through. Use with assigners that
                expect raw logits (e.g. ``DynamicSoftLabelAssigner``).
        prior_format (str): Spatial layout of ``pred_instances.priors`` the
            assigner expects. One of:

              - ``'xyxy'`` (default): pass RPN proposals as-is. Use with
                ``MaxIoUAssigner`` / ``TaskAlignedAssigner`` /
                ``HungarianAssigner`` — they compute centers from
                ``(x1+x2)/2`` internally.
              - ``'point'``: convert each proposal to
                ``(cx, cy, min(w,h), min(w,h))``. Required by
                ``SimOTAAssigner`` / ``DynamicSoftLabelAssigner``, which
                read ``priors[:, :2]`` as the center point and
                ``priors[:, 2]`` as a per-prior stride.
        use_iou_soft_target (bool): If True, the per-positive IoU stored in
            ``AssignResult.max_overlaps`` is forwarded to ``loss_cls`` as a
            soft classification target. Requires ``loss_cls`` to accept a
            ``(labels, iou)`` tuple, i.e. ``QualityFocalLoss``. Defaults to
            False.
    """

    def __init__(self,
                 cls_score_activation: str = 'softmax',
                 prior_format: str = 'xyxy',
                 use_iou_soft_target: bool = False,
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

    def _activate_cls_score(self, cls_score: Tensor) -> Tensor:
        """Map raw cls logits to per-class scores expected by the assigner.

        The output always has shape ``(N, num_classes)``; if the bbox head
        emits a trailing bg channel (softmax-style) it is dropped.
        """
        num_classes = self.bbox_head.num_classes
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
        """Convert XYXY boxes into ``(cx, cy, stride, stride)`` point-prior
        format expected by SimOTA / DynamicSoftLabel.

        ``stride`` is set to ``min(w, h)`` so that the assigner's
        ``center_radius * stride`` zone scales with each proposal's own
        size — important for tiny-object datasets where proposals vary by
        an order of magnitude.
        """
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1.0)
        h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1.0)
        stride = torch.minimum(w, h)
        return torch.stack([cx, cy, stride, stride], dim=-1)

    def _decode_for_assignment(self, priors: Tensor, bbox_pred: Tensor,
                               scores: Tensor) -> Tensor:
        """Decode the regression output into absolute boxes for the assigner.

        Class-agnostic heads emit ``(N, 4)`` and decode directly. Class-aware
        heads emit ``(N, num_classes * 4)``; we pick the regression branch
        corresponding to the highest-scoring foreground class per prior.
        """
        if self.bbox_head.reg_class_agnostic:
            return self.bbox_head.bbox_coder.decode(priors, bbox_pred)

        n = bbox_pred.size(0)
        c = self.bbox_head.num_classes
        bbox_pred = bbox_pred.view(n, c, -1)
        pred_cls = scores.argmax(dim=-1)
        chosen = bbox_pred[torch.arange(n, device=bbox_pred.device), pred_cls]
        return self.bbox_head.bbox_coder.decode(priors, chosen)

    @torch.no_grad()
    def _bbox_forward_for_assignment(
            self, x: Tuple[Tensor],
            proposals: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Run a no-grad bbox forward over all proposals.

        Used purely to populate ``pred_instances`` for the assigner; the
        returned tensors must not feed into the training loss.
        """
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        return bbox_results['cls_score'], bbox_results['bbox_pred']

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Two-pass training: forward all proposals for assignment, then
        forward the sampled subset for loss."""
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        num_imgs = len(batch_data_samples)
        for rpn_results in rpn_results_list:
            if 'bboxes' in rpn_results:
                rpn_results.priors = rpn_results.pop('bboxes')

        proposals = [r.priors for r in rpn_results_list]
        num_proposals_per_img = [p.shape[0] for p in proposals]

        cls_score_all, bbox_pred_all = self._bbox_forward_for_assignment(
            x, proposals)
        cls_score_per_img = cls_score_all.split(num_proposals_per_img, 0)
        bbox_pred_per_img = bbox_pred_all.split(num_proposals_per_img, 0)

        sampling_results: List[SamplingResult] = []
        pos_iou_targets_list: List[Tensor] = []

        for i in range(num_imgs):
            priors_i = proposals[i]
            cls_score_i = cls_score_per_img[i]
            bbox_pred_i = bbox_pred_per_img[i]

            scores_i = self._activate_cls_score(cls_score_i)
            decoded_i = self._decode_for_assignment(priors_i, bbox_pred_i,
                                                    scores_i)
            decoded_i = get_box_tensor(decoded_i)

            # Assigner-facing pred_instances: priors may be point-format.
            assign_priors = (self._xyxy_to_point_priors(priors_i)
                             if self.prior_format == 'point' else priors_i)
            pred_for_assign = InstanceData()
            pred_for_assign.priors = assign_priors
            pred_for_assign.bboxes = decoded_i
            pred_for_assign.scores = scores_i

            assign_result = self.bbox_assigner.assign(
                pred_for_assign, batch_gt_instances[i],
                batch_gt_instances_ignore[i])

            # Sampler-facing pred_instances must keep XYXY priors so that
            # downstream bbox2roi / loss receives valid boxes (and any
            # add_gt_as_proposals concatenation stays well-typed).
            pred_for_sample = InstanceData()
            pred_for_sample.priors = priors_i
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                pred_for_sample,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

            # max_overlaps was extended by sampler.sample() when
            # add_gt_as_proposals=True (GTs prepended with IoU=1), so
            # indexing by pos_inds works for both code paths.
            pos_iou_targets_list.append(
                assign_result.max_overlaps[sampling_result.pos_inds])

        losses = dict()
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results,
                                          pos_iou_targets_list)
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def bbox_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        pos_iou_targets_list: Optional[List[Tensor]] = None,
    ) -> dict:
        """Forward sampled priors and compute bbox losses.

        Falls back to :meth:`StandardRoIHead.bbox_loss` semantics when
        ``use_iou_soft_target`` is False.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        if not self.use_iou_soft_target or pos_iou_targets_list is None:
            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
            bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
            return bbox_results

        labels, label_weights, bbox_targets, bbox_weights = \
            self.bbox_head.get_targets(
                sampling_results, self.train_cfg, concat=True)

        # _get_targets_single lays out each image's samples as
        # [pos..., neg...], and get_targets concatenates across the batch.
        # We mirror that ordering when populating iou_target.
        iou_target = cls_score.new_zeros(labels.shape[0])
        offset = 0
        for samp, pos_ious in zip(sampling_results, pos_iou_targets_list):
            num_pos = samp.pos_inds.numel()
            num_neg = samp.neg_inds.numel()
            if num_pos > 0:
                iou_target[offset:offset + num_pos] = pos_ious
            offset += num_pos + num_neg
        assert offset == labels.shape[0], (
            f'iou_target layout mismatch: filled {offset} of {labels.shape[0]}'
        )

        losses = self._loss_with_iou_target(cls_score, bbox_pred, rois,
                                            labels, label_weights,
                                            bbox_targets, bbox_weights,
                                            iou_target)
        bbox_results.update(loss_bbox=losses)
        return bbox_results

    def _loss_with_iou_target(self, cls_score: Tensor, bbox_pred: Tensor,
                              rois: Tensor, labels: Tensor,
                              label_weights: Tensor, bbox_targets: Tensor,
                              bbox_weights: Tensor,
                              iou_target: Tensor) -> dict:
        """Replicates :meth:`BBoxHead.loss` but passes a ``(labels, iou)``
        tuple to ``loss_cls`` (the contract used by ``QualityFocalLoss``).

        The regression branch is identical to the standard path.
        """
        bh = self.bbox_head
        losses: dict = dict()

        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(
                torch.sum(label_weights > 0).float().item(), 1.)
            loss_cls_ = bh.loss_cls(
                cls_score, (labels, iou_target),
                label_weights,
                avg_factor=avg_factor)
            if isinstance(loss_cls_, dict):
                losses.update(loss_cls_)
            else:
                losses['loss_cls'] = loss_cls_
            losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = bh.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if bh.reg_decoded_bbox:
                    bbox_pred = bh.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if bh.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), bh.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = bh.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0))
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses
