# Copyright (c) OpenMMLab. All rights reserved.
"""RoI head that carries the extra ``iou_pred`` of :class:`IoUAwareShared2FCBBoxHead`.

Only the bbox plumbing differs from :class:`StandardRoIHead`: ``_bbox_forward``
unpacks the 3-tuple ``(cls_score, bbox_pred, iou_pred)``, ``bbox_loss`` forwards
``iou_pred`` to ``loss_and_target``, and ``predict_bbox`` splits ``iou_pred`` per
image so the IoU quality can be fused into the inference score. Everything else
(sampling, cls/bbox losses, mask path) is inherited unchanged.
"""
from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.utils import empty_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class IoUAwareRoIHead(StandardRoIHead):
    """StandardRoIHead variant for :class:`IoUAwareShared2FCBBoxHead`."""

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, iou_pred = self.bbox_head(bbox_feats)
        return dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            iou_pred=iou_pred,
            bbox_feats=bbox_feats)

    def bbox_loss(self, x: Tuple[Tensor], sampling_results: list) -> dict:
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            iou_pred=bbox_results['iou_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        iou_preds = bbox_results['iou_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)
        iou_preds = iou_preds.split(num_proposals_per_img, 0)

        if bbox_preds is not None:
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            iou_preds=iou_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list
