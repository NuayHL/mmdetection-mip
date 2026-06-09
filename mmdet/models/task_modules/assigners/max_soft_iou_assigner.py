# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner


@TASK_UTILS.register_module()
class MaxSoftIoUAssigner(MaxIoUAssigner):
    """MaxIoUAssigner that carries the area-refine **soft-label** mechanism.

    The hard pos/neg assignment is *identical* to :class:`MaxIoUAssigner`
    (IoU thresholding + optional low-quality matching). The only addition
    is that, for every positive sample, the IoU stored in
    ``AssignResult.max_overlaps`` is calibrated by a per-GT reliability
    ratio ``rho`` so that **small objects receive a lifted soft-label
    ceiling**::

        rho = area / (area + r_ref**2)
          pow:    soft_iou = iou ** rho
          add_1:  soft_iou = iou + (1 - rho) * iou * (1 - iou)

    This is the same calibration used by
    :class:`DynamicSoftLabelAssignerAreaRefine`, but applied on top of a
    plain MaxIoU assignment instead of the dynamic-cost matching pipeline.

    The calibrated ``max_overlaps`` is only *consumed* if the RoI head
    forwards it to the classification loss as a soft target — i.e. pair
    this assigner with ``DynAssignRoIHead(use_iou_soft_target=True)`` and a
    ``QualityFocalLoss`` cls head. With a vanilla ``StandardRoIHead`` the
    calibration is inert (max_overlaps is not used by the CE loss /
    ``RandomSampler``), so this assigner is a safe drop-in replacement for
    ``MaxIoUAssigner`` in either case.

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        r_ref (float): Reference object side length (pixels). The transition
            point where ``rho = 0.5`` (i.e. area = r_ref**2). Defaults 32.0.
        calibrate_mode (str): ``'pow'`` → ``iou ** rho``,
            ``'add_1'`` → ``iou + (1 - rho) * iou * (1 - iou)``.
            Defaults to ``'add_1'``.
        soft_label (bool): If False, behaves exactly like MaxIoUAssigner
            (no calibration). Useful as an ablation. Defaults to True.
        **kwargs: Forwarded to :class:`MaxIoUAssigner` (``min_pos_iou``,
            ``match_low_quality``, ``gpu_assign_thr``, ``iou_calculator`` …).
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: Union[float, tuple],
                 r_ref: float = 32.0,
                 calibrate_mode: str = 'add_1',
                 soft_label: bool = True,
                 **kwargs):
        super().__init__(
            pos_iou_thr=pos_iou_thr, neg_iou_thr=neg_iou_thr, **kwargs)
        assert calibrate_mode in ('pow', 'add_1'), \
            f'calibrate_mode must be "pow" or "add_1", got {calibrate_mode!r}'
        self.r_ref = r_ref
        self.calibrate_mode = calibrate_mode
        self.soft_label = soft_label

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Run MaxIoU assignment, then calibrate positive ``max_overlaps``.

        Args:
            pred_instances (:obj:`InstanceData`): Predictions, with
                ``priors`` of shape (n, 4).
            gt_instances (:obj:`InstanceData`): Ground truth, with
                ``bboxes`` (k, 4) and ``labels`` (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Ignored GTs.

        Returns:
            :obj:`AssignResult`: assignment with soft-calibrated
            ``max_overlaps`` on the positive entries.
        """
        assign_result = super().assign(pred_instances, gt_instances,
                                       gt_instances_ignore, **kwargs)
        if not self.soft_label:
            return assign_result

        gt_bboxes = gt_instances.bboxes
        num_gt = gt_bboxes.size(0)
        if num_gt == 0:
            return assign_result

        # Per-GT reliability ratio rho = area / (area + r_ref**2).
        if isinstance(gt_bboxes, BaseBoxes):
            gt_areas = gt_bboxes.areas
        else:
            gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
            gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
            gt_areas = gt_w * gt_h
        rho = gt_areas / (gt_areas + self.r_ref ** 2)  # (num_gt,)

        gt_inds = assign_result.gt_inds
        max_overlaps = assign_result.max_overlaps
        pos_mask = gt_inds > 0
        if pos_mask.any():
            # 0-based gt index per positive sample.
            pos_gt_idx = gt_inds[pos_mask] - 1
            rho_pos = rho.to(max_overlaps.device)[pos_gt_idx]
            iou_pos = max_overlaps[pos_mask].clamp(min=0.0, max=1.0)
            if self.calibrate_mode == 'pow':
                cal = iou_pos.pow(rho_pos)
            else:  # 'add_1'
                cal = iou_pos + (1.0 - rho_pos) * iou_pos * (1.0 - iou_pos)
            max_overlaps[pos_mask] = cal
            assign_result.max_overlaps = max_overlaps

        return assign_result
