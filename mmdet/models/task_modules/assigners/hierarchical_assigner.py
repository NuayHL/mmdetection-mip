# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class HieAssigner(BaseAssigner):
    """Hierarchical Label Assigner (HLA) for RFLA.

    Implements a two-stage label assignment strategy that uses a distance
    metric (e.g. KLD, WD) to measure the affinity between proposals and
    ground-truth boxes, then selects top-k proposals for each GT in a
    hierarchical manner.

    Args:
        gt_max_assign_all (bool): Whether to assign all bboxes with the
            same highest overlap with some gt to that gt. Default True.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes.
            Default -1.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            ``bboxes`` and ``gt_bboxes_ignore``, or the contrary.
            Default True.
        gpu_assign_thr (int): Upper bound of GT count for GPU assign.
            Default -1.
        iou_calculator (dict): Config of IoU/distance calculator.
            Default dict(type='BboxOverlaps2D').
        assign_metric (str): Distance metric for label assignment.
            'kl' (default), 'wd', 'exp_kl', 'kl_10'.
        topk (list[int]): [k1, k2] for two-stage assignment.
            Default [2, 1].
        ratio (float): Anchor rescaling factor for second-stage
            assignment. Default 1.0.
        inside (bool): Whether to filter out proposals whose centers
            fall outside the GT box. Default False.
    """

    def __init__(self,
                 gt_max_assign_all: bool = True,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 gpu_assign_thr: int = -1,
                 iou_calculator: dict = dict(type='BboxOverlaps2D'),
                 assign_metric: str = 'kl',
                 topk: list = [2, 1],  # noqa: B006
                 ratio: float = 1.0,
                 inside: bool = False):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.ratio = ratio
        self.inside = inside

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to bboxes using hierarchical label assignment.

        This method implements the two-stage RFLA assignment:
        1. First stage: for each GT, select top-k1 anchors based on the
           configured distance metric (e.g. KLD).
        2. Second stage: rescale anchors by ``ratio``, recompute metric,
           select top-k2 anchors while preserving first-stage positives.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors`` with shape (n, 4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It includes ``bboxes`` with shape (k, 4),
                and ``labels`` with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        bboxes = pred_instances.priors
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False

        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        # Compute KLD/overlap between GTs and anchors
        overlaps = self.iou_calculator(
            gt_bboxes, bboxes, mode=self.assign_metric)

        # Rescale anchors for second-stage assignment
        bboxes2 = self.anchor_rescale(bboxes, self.ratio)
        overlaps2 = self.iou_calculator(
            gt_bboxes, bboxes2, mode=self.assign_metric)

        # Handle ignored GTs
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        k1 = self.topk[0]
        k2 = self.topk[1]

        # Two-stage hierarchical assignment
        assigned_gt_inds = self.assign_wrt_ranking(
            overlaps, k1, gt_labels)
        assign_result = self.reassign_wrt_ranking(
            assigned_gt_inds, overlaps2, k2, gt_labels)

        # Optional: filter out proposals whose centers are outside GT boxes
        if self.inside:
            num_anchors = bboxes.size(0)
            num_gts = gt_bboxes.size(0)

            anchor_cx = (bboxes[..., 0] + bboxes[..., 2]) / 2
            anchor_cy = (bboxes[..., 1] + bboxes[..., 3]) / 2
            ext_gt_bboxes = gt_bboxes[:, None, :].expand(
                num_gts, num_anchors, 4)
            left = anchor_cx - ext_gt_bboxes[..., 0]
            right = ext_gt_bboxes[..., 2] - anchor_cx
            top = anchor_cy - ext_gt_bboxes[..., 1]
            bottom = ext_gt_bboxes[..., 3] - anchor_cy

            bbox_targets = torch.stack((left, top, right, bottom), -1)
            inside_flag = bbox_targets.min(-1)[0] > 0
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[
                (assign_result.gt_inds - 1).clamp(min=0), length]
            assign_result.gt_inds *= inside_mask

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = \
                assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        return assign_result

    def assign_wrt_ranking(self, overlaps, k, gt_labels=None):
        """First-stage assignment: for each GT, select its top-k anchors.

        Args:
            overlaps (Tensor): Overlap/affinity matrix of shape
                (num_gts, num_bboxes).
            k (int): Number of anchors to select per GT.
            gt_labels (Tensor, optional): GT labels of shape (num_gts,).

        Returns:
            Tensor: Assigned GT indices of shape (num_bboxes,). -1 means
                negative, 0 means background, >0 means positive (1-based).
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        assigned_gt_inds = overlaps.new_full(
            (num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return assigned_gt_inds

        # For each anchor, which GT gives the max overlap
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # For each GT, topk anchors
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(
            k, dim=1, largest=True, sorted=True)

        # Mark anchors with max_overlap < 0.8 as background (0)
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        # Assign top-k anchors for each GT
        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        return assigned_gt_inds

    def reassign_wrt_ranking(self, assign_result, overlaps, k,
                             gt_labels=None):
        """Second-stage refinement assignment.

        Recomputes top-k on rescaled anchors while preserving previously
        assigned positives.

        Args:
            assign_result (Tensor): First-stage assigned GT indices of
                shape (num_bboxes,).
            overlaps (Tensor): Overlap/affinity matrix of shape
                (num_gts, num_bboxes) computed on rescaled anchors.
            k (int): Number of anchors to select per GT.
            gt_labels (Tensor, optional): GT labels of shape (num_gts,).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        mask1 = assign_result <= 0  # background / unassigned
        mask2 = assign_result > 0   # positives from first stage

        assigned_gt_inds = overlaps.new_full(
            (num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = None
            if gt_labels is not None:
                assigned_labels = overlaps.new_full(
                    (num_bboxes,), -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps,
                labels=assigned_labels)

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(
            k, dim=1, largest=True, sorted=True)

        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        # Assign top-k anchors for each GT
        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        # Merge: preserve first-stage positives, fill rest with second-stage
        assigned_gt_inds = assigned_gt_inds * mask1 + assign_result * mask2

        assigned_labels = None
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full(
                (num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps,
            labels=assigned_labels)

    @staticmethod
    def anchor_rescale(bboxes, ratio):
        """Rescale anchor boxes by ratio around their centers.

        Args:
            bboxes (Tensor): Bounding boxes of shape (n, 4) in
                <x1, y1, x2, y2> format.
            ratio (float): Scale factor. ratio=1 means no change,
                ratio<1 means shrink, ratio>1 means expand.

        Returns:
            Tensor: Rescaled bounding boxes.
        """
        center_x = (bboxes[..., 2] + bboxes[..., 0]) / 2
        center_y = (bboxes[..., 3] + bboxes[..., 1]) / 2
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        bboxes[..., 0] = center_x - w * ratio / 2
        bboxes[..., 1] = center_y - h * ratio / 2
        bboxes[..., 2] = center_x + w * ratio / 2
        bboxes[..., 3] = center_y + h * ratio / 2
        return bboxes
