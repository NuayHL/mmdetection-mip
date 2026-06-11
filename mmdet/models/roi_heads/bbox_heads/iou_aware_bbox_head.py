# Copyright (c) OpenMMLab. All rights reserved.
"""IoU-aware Shared2FC bbox head (decoupled localization-quality branch).

This is the *decoupled* placement of the USAA soft-IoU idea, designed to
avoid the failure mode shared by ``SoftLabelRPNHead`` and
``SoftLabelRoIHead``: those put the soft (IoU) target **on the very logit
that is also the test-time ranking score**, which lowers the score of true
positives and suppresses recall/ranking.

Here instead:
  * The original classification head is left **byte-for-byte untouched**
    (softmax + CrossEntropy, ``num_classes + 1`` channels). So this head is
    structurally comparable to the Faster R-CNN baseline, and the ranking
    score of a true positive is never lowered.
  * A **separate scalar IoU branch** (``fc_iou``) taps the regression
    feature and predicts the area-refine calibrated localization quality of
    each positive RoI, supervised with a sigmoid BCE against
    ``IoU(decoded_pred, gt)``.
  * At inference the final ranking score becomes
    ``score = cls_score * sigmoid(iou_pred) ** alpha`` — i.e. the soft IoU
    signal is *added* to the ranking score (which currently carries zero
    localization information) rather than replacing the cls target.

Must be used with :class:`IoUAwareRoIHead`, which plumbs the extra
``iou_pred`` through the train/predict paths.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from .convfc_bbox_head import Shared2FCBBoxHead


@MODELS.register_module()
class IoUAwareShared2FCBBoxHead(Shared2FCBBoxHead):
    """Shared2FCBBoxHead + a decoupled IoU-quality prediction branch.

    Args:
        iou_alpha (float): Exponent on the predicted IoU when fusing it into
            the inference score: ``score = cls * iou_pred ** alpha``.
            Default 0.5.
        r_ref (float): Area-refine reference side length. ``rho = area /
            (area + r_ref ** 2)``. Default 32.0.
        calibrate_mode (str): ``'add_1'`` (default) or ``'pow'``.
        calibrate (bool): If True (default), the IoU *target* is area-refine
            calibrated (lifts the small-object quality ceiling). If False the
            branch regresses the raw IoU — an ablation.
        loss_iou_weight (float): Weight of the IoU branch BCE loss.
            Default 1.0.
    """

    def __init__(self,
                 *args,
                 iou_alpha: float = 0.5,
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'add_1',
                 calibrate: bool = True,
                 loss_iou_weight: float = 1.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert calibrate_mode in ('add_1', 'pow'), calibrate_mode
        self.iou_alpha = iou_alpha
        self.r_ref = r_ref
        self.calibrate_mode = calibrate_mode
        self.calibrate = calibrate
        self.loss_iou_weight = loss_iou_weight
        # Localization-quality branch taps the regression feature.
        self.fc_iou = nn.Linear(self.reg_last_dim, 1)
        nn.init.normal_(self.fc_iou.weight, 0.0, 0.01)
        nn.init.constant_(self.fc_iou.bias, 0.0)

    # ── forward: same as Shared2FCBBoxHead but also emit iou_pred ──────────

    def forward(self, x: Tuple[Tensor]) -> tuple:
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        iou_pred = self.fc_iou(x_reg)  # (N, 1) logits
        return cls_score, bbox_pred, iou_pred

    # ── quality / calibration helpers ─────────────────────────────────────

    @staticmethod
    def _aligned_iou(pred: Tensor, gt: Tensor) -> Tensor:
        eps = 1e-6
        lt = torch.max(pred[:, :2], gt[:, :2])
        rb = torch.min(pred[:, 2:], gt[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * \
            (pred[:, 3] - pred[:, 1]).clamp(min=0)
        area_g = (gt[:, 2] - gt[:, 0]).clamp(min=0) * \
            (gt[:, 3] - gt[:, 1]).clamp(min=0)
        return overlap / (area_p + area_g - overlap + eps)

    def _calibrate(self, q: Tensor, gt: Tensor) -> Tensor:
        q = q.clamp(min=0.0, max=1.0)
        if not self.calibrate:
            return q
        w = (gt[:, 2] - gt[:, 0]).clamp(min=1.0)
        h = (gt[:, 3] - gt[:, 1]).clamp(min=1.0)
        area = w * h
        rho = area / (area + self.r_ref ** 2)
        if self.calibrate_mode == 'pow':
            return q.pow(rho)
        return q + (1.0 - rho) * q * (1.0 - q)

    # ── loss: original cls/bbox loss + decoupled IoU-branch loss ───────────

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        iou_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        # Original (unchanged) cls + bbox losses.
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)
        # Extra decoupled IoU-branch loss.
        labels, _, bbox_targets, _ = cls_reg_targets
        losses['loss_iou'] = self._loss_iou(iou_pred, bbox_pred, rois, labels,
                                            bbox_targets)
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def _loss_iou(self, iou_pred: Tensor, bbox_pred: Tensor, rois: Tensor,
                  labels: Tensor, bbox_targets: Tensor) -> Tensor:
        bg_class_ind = self.num_classes
        pos_mask = (labels >= 0) & (labels < bg_class_ind)
        if bbox_pred is None or not pos_mask.any():
            return iou_pred.sum() * 0.0
        pos = pos_mask.nonzero(as_tuple=False).squeeze(1)
        pos_rois = rois[pos, 1:]

        # decoded predicted box for the assigned class
        if self.reg_class_agnostic:
            pos_delta = bbox_pred.view(bbox_pred.size(0), -1)[pos]
        else:
            pos_delta = bbox_pred.view(
                bbox_pred.size(0), self.num_classes, -1)[pos, labels[pos]]
        pos_pred_box = get_box_tensor(
            self.bbox_coder.decode(pos_rois, pos_delta)).detach()

        # recover the GT box: targets are encoded deltas unless reg_decoded
        if self.reg_decoded_bbox:
            pos_gt_box = get_box_tensor(bbox_targets[pos])
        else:
            pos_gt_box = get_box_tensor(
                self.bbox_coder.decode(pos_rois, bbox_targets[pos]))

        iou = self._aligned_iou(pos_pred_box, pos_gt_box)
        target = self._calibrate(iou, pos_gt_box).detach()
        pred = iou_pred[pos].squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='mean')
        return loss * self.loss_iou_weight

    # ── inference: fuse IoU quality into the ranking score ─────────────────

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        iou_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> list:
        result_list = []
        for img_id in range(len(batch_img_metas)):
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                iou_pred=iou_preds[img_id],
                img_meta=batch_img_metas[img_id],
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                roi: Tensor,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                iou_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = False,
                                rcnn_test_cfg: Optional[ConfigDict] = None):
        """Identical to the parent, but multiplies the foreground scores by
        ``sigmoid(iou_pred) ** iou_alpha`` before NMS."""
        from mmdet.models.layers import multiclass_nms
        from mmdet.models.utils import empty_instances
        from mmdet.structures.bbox import scale_boxes
        from mmengine.structures import InstanceData

        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        # ---- fuse decoupled IoU quality into the ranking score ----
        if scores is not None and iou_pred is not None:
            iou_factor = iou_pred.sigmoid().clamp(min=0.0, max=1.0)
            iou_factor = iou_factor.pow(self.iou_alpha)  # (N, 1)
            # foreground columns only (bg column, if any, is unused for det)
            scores = scores.clone()
            scores[:, :self.num_classes] = \
                scores[:, :self.num_classes] * iou_factor

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
