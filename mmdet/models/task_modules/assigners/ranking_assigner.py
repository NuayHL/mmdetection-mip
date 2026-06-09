# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class RankingAssigner(BaseAssigner):
    """Ranking-based K Assignment (RKA) for NWD-RKA.

    For each gt box, assign the top-k proposals (ranked by a similarity
    metric such as NWD) as positive samples. The remaining proposals are
    assigned a negative label.

    Each proposal is assigned with ``-1``, ``0`` or a positive integer:

    - -1: ignored sample (not selected as positive, similarity too high
      to be negative).
    - 0: negative sample, no assigned gt.
    - positive integer: positive sample, the 1-based index of the
      assigned gt.

    Args:
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            ``gt_instances_ignore`` is specified). Negative values mean not
            ignoring any bboxes. Default -1.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            ``bboxes`` and ``gt_bboxes_ignore``, or the contrary.
            Default True.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will
            assign on CPU device. Negative values mean not assign on CPU.
            Default -1.
        iou_calculator (dict): Config of the similarity calculator, e.g.
            ``dict(type='BboxDistanceMetric')``. Default
            ``dict(type='BboxOverlaps2D')``.
        assign_metric (str): The metric of measuring the similarity between
            boxes, e.g. 'nwd', 'iou'. Default 'iou'.
        topk (int): Assign k positive samples to each gt. Default 1.
    """

    def __init__(self,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 gpu_assign_thr: int = -1,
                 iou_calculator: dict = dict(type='BboxOverlaps2D'),
                 assign_metric: str = 'iou',
                 topk: int = 1):
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to bboxes using ranking-based k assignment.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors`` with shape (n, 4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It includes ``bboxes`` with shape (k, 4) and
                ``labels`` with shape (k, ).
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
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(
            gt_bboxes, bboxes, mode=self.assign_metric)

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

        assign_result = self.assign_wrt_ranking(overlaps, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self, overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(
            self.topk, dim=1, largest=True, sorted=True)

        # pre-assign negative samples
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.3)] = 0

        # assign positive samples to each gt wrt ranking
        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
