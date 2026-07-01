# Copyright (c) OpenMMLab. All rights reserved.
"""DINOHeadUSAA — DINO head with USAA-style calibrated soft-label supervision.

Faithful mmdetection port of USAA "Change 2" from the RT-DETR flavour
(``RTDETRDetectionLoss_USAA._calibrate_soft_label`` in
``Ultralytics/ultralytics/models/utils/loss.py``).

Background
----------
mmdetection's :class:`DINOHead` already supports an IoU soft-label
classification target: when ``loss_cls`` is a :class:`QualityFocalLoss`, both
:meth:`DETRHead.loss_by_feat_single` and :meth:`DINOHead._loss_dn_single`
supervise each *positive* query with

    score = IoU(decoded_pred_box, matched_gt_box)          # ∈ [0, 1]

instead of a hard 1.  This is the mmdet analogue of RT-DETR's VFL
(``use_vfl=True``) IoU soft label.

USAA modification
-----------------
Small objects rarely reach a high IoU even when correctly detected, so a raw-IoU
soft label systematically under-supervises them.  USAA calibrates the target
with a per-GT, size-dependent lift:

    ρ_i         = area_i / (area_i + r_ref_cal²)            ∈ (0, 1]
    add_1:  f(u,ρ) = u + (1 − ρ) · u · (1 − u)             (Bernoulli-variance)
    pow:    f(u,ρ) = u^ρ

``area_i`` is in **pixels²** (GT boxes rescaled to absolute image coords), so
``r_ref_cal`` is a pixel reference — the same ``ρ = area / (area + r_ref²)``
convention used by every other USAA component in this repo.  The lift is largest
for small objects (ρ → 0) and vanishes for large ones (ρ → 1); it is also small
early in training (u ≈ 0, safe), maximal at intermediate IoU, and decays to 0 as
u → 1 (so it never pushes a target above 1).

This calibration is applied to the matching-query decoder losses, the encoder
(two-stage proposal) loss — both routed through
:meth:`loss_by_feat_single` — and the denoising (DN) loss.  Setting
``use_soft_label_cal=False`` recovers :class:`DINOHead` exactly (raw IoU soft
label), for an A/B on the calibrated *supervision signal* alone.

Requires ``loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, ...)``.
With the default ``FocalLoss`` there is no IoU target to calibrate and this head
behaves identically to :class:`DINOHead`.
"""
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, reduce_mean
from ..losses import QualityFocalLoss
from .dino_head import DINOHead


@MODELS.register_module()
class DINOHeadUSAA(DINOHead):
    """DINO head with area-refine calibrated IoU soft labels.

    Args:
        use_soft_label_cal (bool): Enable the USAA soft-label calibration. If
            ``False``, behaves exactly like :class:`DINOHead`. Defaults to True.
        r_ref_cal (float): Pixel reference size for the calibration ratio
            ``ρ = area / (area + r_ref_cal²)``. Defaults to 32.0.
        cal_type (str): ``'add_1'`` (Bernoulli-variance lift) or ``'pow'``
            (``u**ρ``). Defaults to ``'add_1'``.
        calibrate_dn (bool): Also calibrate the denoising (DN) IoU soft label.
            Defaults to True (matches RT-DETR, whose DN shares ``_get_loss``).
    """

    def __init__(self,
                 *args,
                 use_soft_label_cal: bool = True,
                 r_ref_cal: float = 32.0,
                 cal_type: str = 'add_1',
                 calibrate_dn: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_soft_label_cal = use_soft_label_cal
        self.r_ref_cal = float(r_ref_cal)
        self.cal_type = cal_type
        self.calibrate_dn = calibrate_dn
        assert cal_type in ('add_1', 'pow', 'none'), \
            f"cal_type must be 'add_1'|'pow'|'none', got {cal_type}"

    # ------------------------------------------------------------------ #
    # USAA soft-label calibration
    # ------------------------------------------------------------------ #
    def _calibrate_iou(self, iou: Tensor, gt_bboxes_px: Tensor) -> Tensor:
        """Area-refine calibration of the per-positive IoU soft label.

        Args:
            iou (Tensor): (num_pos,) raw IoU between decoded prediction and its
                matched GT box.
            gt_bboxes_px (Tensor): (num_pos, 4) matched GT boxes in **absolute
                pixel** ``xyxy``.

        Returns:
            Tensor: (num_pos,) calibrated IoU, clamped to [0, 1].
        """
        if not self.use_soft_label_cal or self.cal_type == 'none' \
                or iou.numel() == 0:
            return iou
        w = (gt_bboxes_px[:, 2] - gt_bboxes_px[:, 0]).clamp(min=0)
        h = (gt_bboxes_px[:, 3] - gt_bboxes_px[:, 1]).clamp(min=0)
        area = (w * h).clamp(min=1.0)
        rho = area / (area + self.r_ref_cal ** 2)
        if self.cal_type == 'add_1':
            iou = iou + (1.0 - rho) * iou * (1.0 - iou)
        elif self.cal_type == 'pow':
            iou = iou.pow(rho)
        return iou.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Matching-query + encoder loss (single decoder layer)
    # ------------------------------------------------------------------ #
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Copy of :meth:`DETRHead.loss_by_feat_single` with the QFL IoU soft
        label calibrated by :meth:`_calibrate_iou`.

        Only the ``QualityFocalLoss`` branch differs from the parent; the
        regression losses are untouched.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # construct factors used for rescaling bboxes (needed early for the
        # pixel-space GT area in the soft-label calibration).
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            iou = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            # ★ USAA: calibrate the IoU soft label using pixel-space GT area.
            pos_gt_bboxes_px = pos_decode_bbox_targets * factors[pos_inds]
            scores[pos_inds] = self._calibrate_iou(iou, pos_gt_bboxes_px)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                label_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    # ------------------------------------------------------------------ #
    # Denoising (DN) loss (single decoder layer)
    # ------------------------------------------------------------------ #
    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Copy of :meth:`DINOHead._loss_dn_single` with the QFL IoU soft label
        calibrated by :meth:`_calibrate_iou` (only when ``calibrate_dn``)."""
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # construct factors used for rescaling bboxes (needed early for the
        # pixel-space GT area in the soft-label calibration).
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                bg_class_ind = self.num_classes
                pos_inds = ((labels >= 0)
                            & (labels < bg_class_ind)).nonzero().squeeze(1)
                scores = label_weights.new_zeros(labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
                # ★ USAA: calibrate the DN IoU soft label (pixel-space area).
                if self.calibrate_dn:
                    pos_gt_bboxes_px = pos_decode_bbox_targets * \
                        factors[pos_inds]
                    iou = self._calibrate_iou(iou, pos_gt_bboxes_px)
                scores[pos_inds] = iou
                loss_cls = self.loss_cls(
                    cls_scores, (labels, scores),
                    weight=label_weights,
                    avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
