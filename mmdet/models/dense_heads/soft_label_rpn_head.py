# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from .rpn_head import RPNHead


@MODELS.register_module()
class SoftLabelRPNHead(RPNHead):
    """RPN head with a **soft objectness label** (NWD/IoU quality target).

    This is the RPN-stage analogue of :class:`SoftLabelRoIHead`: the binary
    objectness target of each positive anchor is turned from a hard ``1`` into
    a *soft* quality value, supervised with ``QualityFocalLoss`` (GFL-style).
    It is the faithful two-stage translation of the original one-stage (YOLO)
    soft-label assignment, applied at the RPN — the stage that actually drives
    tiny-object recall.

    **Fusion with NWD-RKA.** The selection of which anchors are positive is
    left to the configured RPN assigner (use ``RankingAssigner`` with the NWD
    metric to reproduce NWD-RKA). This head only changes *how the positives
    are supervised*: the soft target is the **area-refine calibrated NWD**
    between the decoded predicted box and the assigned GT, so the two ideas
    compose along orthogonal axes — NWD as the *metric*, soft label as the
    *supervision*.

    The quality target is computed on the fly in :meth:`loss_by_feat_single`
    (exactly like :class:`GFLHead` does for IoU): for each positive anchor the
    assigned GT box is recovered by decoding ``bbox_targets``, and the quality
    is ``calibrate(metric(decoded_pred, gt))``.

    Args:
        quality_metric (str): ``'nwd'`` (default) or ``'iou'`` — the metric
            used as the soft objectness target.
        constant (float): NWD normalization constant ``C``. Default 12.7.
        nwd_weight (float): NWD wh-distance weight. Default 2.0.
        r_ref (float): Area-refine reference side length. ``rho = area /
            (area + r_ref**2)``. Default 32.0.
        calibrate_mode (str): ``'add_1'`` (default) or ``'pow'``.
        soft_label (bool): If False, positives get a hard target of 1.0
            (QFL degenerates to focal BCE) — an ablation. Default True.
        **kwargs: Forwarded to :class:`RPNHead` (must set ``loss_cls`` to a
            ``QualityFocalLoss`` with ``use_sigmoid=True``).
    """

    def __init__(self,
                 *args,
                 quality_metric: str = 'nwd',
                 constant: float = 12.7,
                 nwd_weight: float = 2.0,
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'add_1',
                 soft_label: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert quality_metric in ('nwd', 'iou'), quality_metric
        assert calibrate_mode in ('add_1', 'pow'), calibrate_mode
        self.quality_metric = quality_metric
        self.constant = constant
        self.nwd_weight = nwd_weight
        self.r_ref = r_ref
        self.calibrate_mode = calibrate_mode
        self.soft_label = soft_label

    # ── quality / calibration helpers ─────────────────────────────────────

    def _aligned_quality(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Per-row quality between aligned ``pred`` and ``gt`` boxes (xyxy)."""
        eps = 1e-6
        if self.quality_metric == 'iou':
            lt = torch.max(pred[:, :2], gt[:, :2])
            rb = torch.min(pred[:, 2:], gt[:, 2:])
            wh = (rb - lt).clamp(min=0)
            overlap = wh[:, 0] * wh[:, 1]
            area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * \
                (pred[:, 3] - pred[:, 1]).clamp(min=0)
            area_g = (gt[:, 2] - gt[:, 0]).clamp(min=0) * \
                (gt[:, 3] - gt[:, 1]).clamp(min=0)
            return overlap / (area_p + area_g - overlap + eps)

        # NWD (aligned), matching BboxDistanceMetric's 'nwd' formula.
        cx1 = (pred[:, 0] + pred[:, 2]) * 0.5
        cy1 = (pred[:, 1] + pred[:, 3]) * 0.5
        cx2 = (gt[:, 0] + gt[:, 2]) * 0.5
        cy2 = (gt[:, 1] + gt[:, 3]) * 0.5
        w1 = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
        h1 = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
        w2 = (gt[:, 2] - gt[:, 0]).clamp(min=eps)
        h2 = (gt[:, 3] - gt[:, 1]).clamp(min=eps)
        center_d = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + eps
        wh_d = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / (self.nwd_weight ** 2)
        wasserstein = torch.sqrt(center_d + wh_d)
        return torch.exp(-wasserstein / self.constant)

    def _calibrate(self, quality: Tensor, gt: Tensor) -> Tensor:
        """Area-refine calibration: lift the soft target for small objects."""
        w = (gt[:, 2] - gt[:, 0]).clamp(min=1.0)
        h = (gt[:, 3] - gt[:, 1]).clamp(min=1.0)
        area = w * h
        rho = area / (area + self.r_ref ** 2)
        q = quality.clamp(min=0.0, max=1.0)
        if self.calibrate_mode == 'pow':
            return q.pow(rho)
        return q + (1.0 - rho) * q * (1.0 - q)

    # ── per-level loss with soft objectness target ────────────────────────

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Same as :meth:`AnchorHead.loss_by_feat_single` but the objectness
        ``loss_cls`` receives a ``(labels, soft_quality)`` tuple for QFL.

        The regression branch is byte-for-byte identical to the parent.
        """
        # ---- classification (QFL with soft NWD/IoU target) ----
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        # flatten reg-related tensors (needed both for the soft target and the
        # regression loss below)
        target_dim = bbox_targets.size(-1)
        anchors_flat = anchors.reshape(-1, anchors.size(-1))
        bbox_targets_flat = bbox_targets.reshape(-1, target_dim)
        bbox_weights_flat = bbox_weights.reshape(-1, target_dim)
        bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_coder.encode_size)

        # RPN: FG label 0, BG label 1 (== num_classes).
        score = label_weights.new_zeros(labels.shape)
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).squeeze(1)
        if pos_inds.numel() > 0:
            if self.soft_label:
                pos_anchors = anchors_flat[pos_inds]
                pos_pred = self.bbox_coder.decode(pos_anchors,
                                                  bbox_pred_flat[pos_inds])
                pos_pred = get_box_tensor(pos_pred).detach()
                # Recover the assigned GT box: decode(anchor, encoded_target).
                pos_gt = self.bbox_coder.decode(pos_anchors,
                                                bbox_targets_flat[pos_inds])
                pos_gt = get_box_tensor(pos_gt)
                quality = self._aligned_quality(pos_pred, pos_gt)
                score[pos_inds] = self._calibrate(quality, pos_gt)
            else:
                score[pos_inds] = 1.0

        loss_cls = self.loss_cls(
            cls_score, (labels, score), label_weights, avg_factor=avg_factor)

        # ---- regression (identical to AnchorHead) ----
        bbox_targets_flat = bbox_targets_flat.reshape(-1, target_dim)
        bbox_weights_flat = bbox_weights_flat.reshape(-1, target_dim)
        if self.reg_decoded_bbox:
            bbox_pred_dec = self.bbox_coder.decode(anchors_flat, bbox_pred_flat)
            bbox_pred_dec = get_box_tensor(bbox_pred_dec)
            loss_bbox = self.loss_bbox(
                bbox_pred_dec, bbox_targets_flat, bbox_weights_flat,
                avg_factor=avg_factor)
        else:
            loss_bbox = self.loss_bbox(
                bbox_pred_flat, bbox_targets_flat, bbox_weights_flat,
                avg_factor=avg_factor)
        return loss_cls, loss_bbox
