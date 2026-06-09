# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.models.losses.accuracy import accuracy
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import get_box_tensor
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class SoftLabelRoIHead(StandardRoIHead):
    """StandardRoIHead + IoU **soft-label** classification target.

    This is the minimal head that turns the per-positive IoU produced by the
    assigner (``AssignResult.max_overlaps`` — area-refine calibrated when the
    assigner is :class:`MaxSoftIoUAssigner`) into a *soft* classification
    target for ``QualityFocalLoss``.

    The only differences from :class:`StandardRoIHead`:

    1. :meth:`loss` additionally records, per image, the IoU of each positive
       sample (``assign_result.max_overlaps[pos_inds]``) and forwards it to
       :meth:`bbox_loss`. The assignment itself is the **standard single-pass
       IoU assignment** — unlike :class:`DynAssignRoIHead` there is **no**
       prediction-aware forward over all proposals, so no extra compute.
    2. :meth:`bbox_loss` passes a ``(labels, iou_target)`` tuple to
       ``loss_cls`` (the ``QualityFocalLoss`` contract) instead of the plain
       hard ``labels``.

    Everything else (RoI extraction, regression branch, sampler, mask head)
    is inherited unchanged. Use with a ``QualityFocalLoss`` cls head, e.g.::

        roi_head=dict(
            type='SoftLabelRoIHead',
            bbox_head=dict(
                loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True,
                              beta=2.0, loss_weight=1.0,
                              custom_cls_channels=True)))
        train_cfg=dict(rcnn=dict(assigner=dict(type='MaxSoftIoUAssigner', ...)))
    """

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Same as :meth:`StandardRoIHead.loss` but also captures the
        per-positive IoU soft target for the classification loss."""
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        num_imgs = len(batch_data_samples)
        sampling_results: List[SamplingResult] = []
        pos_iou_targets_list: List[Tensor] = []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

            # sampler.sample() prepends GTs with max_overlaps=1 when
            # add_gt_as_proposals=True, so indexing by pos_inds is valid for
            # both code paths.
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
        """Forward sampled RoIs and compute the soft-label bbox losses."""
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        if pos_iou_targets_list is None:
            # Fall back to the standard hard-label path.
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

        # _get_targets_single lays each image out as [pos..., neg...] and
        # get_targets concatenates across the batch; mirror that ordering.
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
        tuple to ``loss_cls`` (the ``QualityFocalLoss`` contract). The
        regression branch is identical to the standard path."""
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
