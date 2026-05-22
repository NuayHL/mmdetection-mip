# Copyright (c) OpenMMLab. All rights reserved.
"""FCOS head with Task-Aligned Label Assignment (TAL).

Based on TOOD: Task-aligned One-stage Object Detection.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import NormedConv2d
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..utils import multi_apply
from .fcos_head import FCOSHead

INF = 1e8


@MODELS.register_module()
class FCOSTALHead(FCOSHead):
    """FCOS head with Task-Aligned Learning (TAL) for label assignment.

    This head replaces the default FCOS label assignment (center + regress
    range + min area) with TaskAlignedAssigner from TOOD, which aligns
    classification and localization by selecting top-k anchors per gt
    according to alignment_metric = score^alpha * iou^beta.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config
            with keys: assigner (TaskAlignedAssigner), alpha, beta.
            Defaults to None.
        **kwargs: Other arguments same as FCOSHead.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = False,
                 centerness_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 cls_predictor_cfg=None,
                 train_cfg: ConfigType = dict(
                     assigner=dict(type='TaskAlignedAssigner', topk=13),
                     alpha=1,
                     beta=6),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            regress_ranges=regress_ranges,
            center_sampling=center_sampling,
            center_sample_radius=center_sample_radius,
            norm_on_bbox=norm_on_bbox,
            centerness_on_reg=centerness_on_reg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            norm_cfg=norm_cfg,
            cls_predictor_cfg=cls_predictor_cfg,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            **kwargs)
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.alpha = self.train_cfg.get('alpha', 1)
            self.beta = self.train_cfg.get('beta', 6)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss with TAL assignment."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # TAL needs predictions for assignment: get targets with cls/bbox
        labels, bbox_targets, alignment_metrics = self.get_targets(
            all_level_points,
            batch_gt_instances,
            flatten_cls_scores,
            flatten_bbox_preds,
            flatten_points,
            num_imgs)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_alignment_metrics = torch.cat(alignment_metrics)

        losses = dict()
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        losses['loss_cls'] = loss_cls

        if getattr(self.loss_cls, 'custom_accuracy', False):
            acc = self.loss_cls.get_accuracy(flatten_cls_scores,
                                             flatten_labels)
            losses.update(acc)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_alignment_metrics = flatten_alignment_metrics[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        # Use alignment_metrics to weight bbox loss (TAL), fallback to centerness
        bbox_weight = pos_alignment_metrics
        bbox_denorm = max(
            reduce_mean(bbox_weight.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=bbox_weight,
                avg_factor=bbox_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        losses['loss_bbox'] = loss_bbox
        losses['loss_centerness'] = loss_centerness
        return losses

    def get_targets(
        self,
        points: List[Tensor],
        batch_gt_instances: InstanceList,
        flatten_cls_scores: Tensor,
        flatten_bbox_preds: Tensor,
        flatten_points: Tensor,
        num_imgs: int,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Compute labels, bbox_targets and alignment_metrics using TAL."""
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        num_points_per_lvl = [p.size(0) for p in points]
        num_points_per_img = sum(num_points_per_lvl)

        # Stride per point for decoding (norm_on_bbox)
        strides = self.strides
        if isinstance(strides[0], (list, tuple)):
            strides = [s[0] for s in strides]
        stride_expand = flatten_points.new_zeros(flatten_points.size(0), 4)
        beg = 0
        for i, num_p in enumerate(num_points_per_lvl):
            end = beg + num_p * num_imgs
            stride_expand[beg:end] = strides[i]
            beg = end

        # Decode bbox_pred to (x1,y1,x2,y2) for assigner
        bbox_pred_to_decode = flatten_bbox_preds
        if self.norm_on_bbox:
            bbox_pred_to_decode = bbox_pred_to_decode.clamp(min=0) * stride_expand
        else:
            bbox_pred_to_decode = bbox_pred_to_decode.exp()
        decoded_bbox_preds = self.bbox_coder.decode(
            flatten_points, bbox_pred_to_decode)

        # Priors for TAL: center must be in gt; use (x,y,x,y) so center=(x,y)
        priors_xyxy = torch.stack([
            flatten_points[:, 0], flatten_points[:, 1],
            flatten_points[:, 0], flatten_points[:, 1]
        ], dim=1)

        # Per-image assignment
        labels_list = []
        bbox_targets_list = []
        alignment_metrics_list = []

        for i in range(num_imgs):
            start = i * num_points_per_img
            end = (i + 1) * num_points_per_img
            cls_i = flatten_cls_scores[start:end]  # (N, num_classes)
            bbox_i = decoded_bbox_preds[start:end]  # (N, 4) xyxy
            points_i = flatten_points[start:end]   # (N, 2)
            priors_i = priors_xyxy[start:end]      # (N, 4)

            gt_instances = batch_gt_instances[i]
            pred_instances = InstanceData(
                priors=priors_i,
                scores=torch.sigmoid(cls_i),
                bboxes=bbox_i)
            assign_result = self.assigner.assign(
                pred_instances, gt_instances, None, self.alpha, self.beta)

            assigned_gt_inds = assign_result.gt_inds  # 0=bg, 1-based=gt index
            assigned_labels = assign_result.labels    # -1 for bg, else class id
            assign_metrics = getattr(
                assign_result, 'assign_metrics',
                assign_result.max_overlaps)

            num_pts = points_i.size(0)
            labels = points_i.new_full(
                (num_pts,), self.num_classes, dtype=torch.long)
            bbox_targets = points_i.new_zeros((num_pts, 4))
            alignment_metrics = points_i.new_zeros(num_pts)

            pos_mask = assigned_gt_inds > 0
            if pos_mask.any():
                pos_inds = pos_mask.nonzero().squeeze(1)
                gt_inds = assigned_gt_inds[pos_inds] - 1
                gt_bboxes = gt_instances.bboxes
                gt_labels_assign = assigned_labels[pos_inds]

                labels[pos_inds] = gt_labels_assign
                alignment_metrics[pos_inds] = assign_metrics[pos_inds]

                # Bbox targets: (left, top, right, bottom) from point to gt
                xs = points_i[pos_inds, 0]
                ys = points_i[pos_inds, 1]
                gt_bboxes_pos = gt_bboxes[gt_inds]
                left = xs - gt_bboxes_pos[:, 0]
                top = ys - gt_bboxes_pos[:, 1]
                right = gt_bboxes_pos[:, 2] - xs
                bottom = gt_bboxes_pos[:, 3] - ys
                bbox_targets[pos_inds, :] = torch.stack(
                    [left, top, right, bottom], dim=1)

            labels_list.append(labels)
            bbox_targets_list.append(bbox_targets)
            alignment_metrics_list.append(alignment_metrics)

        # Concat per level (same as original FCOS: batch first, then level)
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_alignment_metrics = []
        for lvl_idx in range(num_levels):
            n_cur = num_points_per_lvl[lvl_idx]
            lvl_labels = []
            lvl_bbox = []
            lvl_align = []
            for img_idx in range(num_imgs):
                base = img_idx * num_points_per_img
                # Level lvl_idx in this image
                lvl_start = base + sum(num_points_per_lvl[:lvl_idx])
                lvl_end = lvl_start + n_cur
                lvl_labels.append(labels_list[img_idx].new_full(
                    (n_cur,), self.num_classes))
                lvl_bbox.append(bbox_targets_list[img_idx].new_zeros(n_cur, 4))
                lvl_align.append(alignment_metrics_list[img_idx].new_zeros(n_cur))
                # Copy from per-img flat (we stored per-img flat)
                src_start = sum(num_points_per_lvl[:lvl_idx])
                src_end = src_start + n_cur
                lvl_labels[-1] = labels_list[img_idx][src_start:src_end]
                lvl_bbox[-1] = bbox_targets_list[img_idx][src_start:src_end]
                lvl_align[-1] = alignment_metrics_list[img_idx][src_start:src_end]
            concat_lvl_labels.append(torch.cat(lvl_labels))
            bbox_cat = torch.cat(lvl_bbox)
            if self.norm_on_bbox:
                bbox_cat = bbox_cat / strides[lvl_idx]
            concat_lvl_bbox_targets.append(bbox_cat)
            concat_lvl_alignment_metrics.append(torch.cat(lvl_align))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_alignment_metrics
