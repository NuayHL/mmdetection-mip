# Copyright (c) OpenMMLab. All rights reserved.
"""HungarianAssignerUSAA — scale-aware one-to-one matcher for DETR-family heads.

A faithful mmdetection port of the RT-DETR ``HungarianMatcher_ScaleAware`` used
by the USAA flavour in Ultralytics
(``Ultralytics/ultralytics/models/utils/mla_detr.py``).

Motivation
----------
When the classification target of small objects is softened (the USAA
soft-label calibration, see :class:`DINOHeadUSAA`), the cls channel becomes a
less discriminative matching cue for those objects.  To keep the one-to-one
assignment well-behaved, the matching *cost* should lean on spatial cues (L1 /
IoU) rather than classification for small objects.  This is done by modulating
the per-GT cost weights with a size-dependent reliability ratio ``ρ``:

    ρ_i               = area_i / (area_i + r_ref_ab²)          ∈ (0, 1]
    mod_i             = 1 − ρ_i                                ∈ [0, 1)   (1 = smallest)
    cost_cls_i        ×= 1 − mod_i · cls_reduction   (cls cost DOWN for small obj)
    cost_spatial_i    ×= 1 + mod_i · spatial_boost   (L1/IoU cost UP for small obj)

``area_i`` is measured in **pixels²** from the (absolute, xyxy) GT boxes, so
``r_ref_ab`` is a pixel reference — identical convention to every other USAA
component in this repo (``ρ = area / (area + r_ref²)``).

Unlike RT-DETR (which hard-codes a ``{class, bbox, giou}`` cost triplet), the
mmdetection :class:`HungarianAssigner` takes an arbitrary ``match_costs`` list.
Each cost module is therefore classified once at construction time into a
``'cls'`` / ``'spatial'`` / ``'other'`` group by its class name (overridable),
and its column-wise (per-GT) contribution is scaled accordingly before the
costs are summed and Hungarian-matched.  Setting ``scale_aware=False`` recovers
the vanilla :class:`HungarianAssigner` exactly (for A/B ablation).
"""
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import get_box_tensor
from .assign_result import AssignResult
from .hungarian_assigner import HungarianAssigner

# Default cost-module class names for each group.  The classification cost is
# reduced for small objects; the spatial (localization) costs are boosted.
_DEFAULT_CLS_COST_TYPES = ('FocalLossCost', 'ClassificationCost',
                           'BinaryFocalLossCost', 'CrossEntropyLossCost')
_DEFAULT_SPATIAL_COST_TYPES = ('BBoxL1Cost', 'IoUCost')


@TASK_UTILS.register_module()
class HungarianAssignerUSAA(HungarianAssigner):
    """Scale-aware :class:`HungarianAssigner` with per-GT dynamic cost weights.

    Args:
        match_costs (dict | ConfigDict | list): Match cost configs, identical to
            :class:`HungarianAssigner` (e.g. ``FocalLossCost`` + ``BBoxL1Cost``
            + ``IoUCost``).
        r_ref_ab (float): Pixel reference size for the cost-modulation ratio
            ``ρ = area / (area + r_ref_ab²)``. Larger → modulation reaches
            bigger objects. Defaults to 64.0.
        cls_reduction (float): Maximum fractional reduction of the classification
            cost weight for the smallest objects (``mod → 1``). ``0`` disables
            cls modulation. Defaults to 0.5.
        spatial_boost (float): Maximum fractional boost of the spatial (L1 / IoU)
            cost weight for the smallest objects. ``0`` disables spatial
            modulation. Defaults to 0.5.
        scale_aware (bool): If ``False``, behave exactly like
            :class:`HungarianAssigner` (no per-GT modulation). Defaults to True.
        cls_cost_types (Sequence[str]): Cost class names treated as the
            classification group. Defaults to ``_DEFAULT_CLS_COST_TYPES``.
        spatial_cost_types (Sequence[str]): Cost class names treated as the
            spatial group. Defaults to ``_DEFAULT_SPATIAL_COST_TYPES``.
    """

    def __init__(self,
                 match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                    ConfigDict],
                 r_ref_ab: float = 64.0,
                 cls_reduction: float = 0.5,
                 spatial_boost: float = 0.5,
                 scale_aware: bool = True,
                 cls_cost_types: Sequence[str] = _DEFAULT_CLS_COST_TYPES,
                 spatial_cost_types: Sequence[str] = _DEFAULT_SPATIAL_COST_TYPES
                 ) -> None:
        super().__init__(match_costs)
        self.r_ref_ab = float(r_ref_ab)
        self.cls_reduction = float(cls_reduction)
        self.spatial_boost = float(spatial_boost)
        self.scale_aware = scale_aware
        self.cls_cost_types = tuple(cls_cost_types)
        self.spatial_cost_types = tuple(spatial_cost_types)

        # Classify each match cost once, by its registered class name.
        self.cost_groups: List[str] = []
        for match_cost in self.match_costs:
            name = type(match_cost).__name__
            if name in self.cls_cost_types:
                self.cost_groups.append('cls')
            elif name in self.spatial_cost_types:
                self.cost_groups.append('spatial')
            else:
                self.cost_groups.append('other')

    def _per_gt_factors(self, gt_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Per-GT multipliers for the cls / spatial cost groups.

        Args:
            gt_bboxes (Tensor): (num_gts, 4) GT boxes in **absolute pixel**
                ``xyxy`` (or a :obj:`BaseBoxes` wrapping the same).

        Returns:
            Tuple[Tensor, Tensor]: ``cls_factor`` and ``spatial_factor``, each
            of shape (num_gts,). ``cls_factor`` ∈ (1−cls_reduction, 1];
            ``spatial_factor`` ∈ [1, 1+spatial_boost).
        """
        gt = get_box_tensor(gt_bboxes)
        w = (gt[:, 2] - gt[:, 0]).clamp(min=0)
        h = (gt[:, 3] - gt[:, 1]).clamp(min=0)
        area = (w * h).clamp(min=1.0)
        rho = area / (area + self.r_ref_ab ** 2)
        mod = 1.0 - rho  # 1 for the smallest objects, → 0 for the largest
        cls_factor = 1.0 - mod * self.cls_reduction
        spatial_factor = 1.0 + mod * self.spatial_boost
        return cls_factor, spatial_factor

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """One-to-one matching with per-GT scale-aware cost weighting.

        Mirrors :meth:`HungarianAssigner.assign` exactly, except that each match
        cost is scaled column-wise (per GT) by ``cls_factor`` or
        ``spatial_factor`` before the costs are summed.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost, with per-GT scale-aware modulation
        if self.scale_aware:
            cls_factor, spatial_factor = self._per_gt_factors(
                gt_instances.bboxes)
        cost_list = []
        for match_cost, group in zip(self.match_costs, self.cost_groups):
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)  # (num_preds, num_gts)
            if self.scale_aware and group != 'other':
                factor = cls_factor if group == 'cls' else spatial_factor
                cost = cost * factor.to(cost.dtype)[None, :]
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
