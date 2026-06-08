# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.registry import TASK_UTILS

DeviceType = Union[str, torch.device]


@TASK_UTILS.register_module()
class RFGenerator:
    """Receptive Field based anchor generator for 2D anchor-based detectors.

    Instead of using fixed base sizes derived from strides, this generator
    computes base anchor sizes from the theoretical receptive field (TRF) of
    a standard ResNet-50-FPN backbone using the Distill RF computation method
    (https://distill.pub/2019/computing-receptive-fields/).

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and
            width of anchors in a single level. Default [0.5, 1.0, 2.0].
        fraction (float): Fraction of TRF used as effective receptive
            field (ERF). Default 0.5.
        fpn_layer (str): The start level of FPN for anchor generation.
            'p3' for single-stage detectors (uses P3-P7 TRFs),
            'p2' for two-stage detectors (uses P2-P6 TRFs).
            Default 'p3'.
        scales (list[float]): Anchor scales for anchors in a single level.
            Default [1.0].
        base_sizes (list[int], optional): Basic sizes of anchors in
            multiple levels. If None, computed from TRF.
        scale_major (bool): Whether to multiply scales first when
            generating base anchors. Default True.
        centers (list[tuple[float]], optional): The centers of the anchor
            relative to the feature grid center.
        center_offset (float): The offset of center in proportion to
            anchors' width and height. Default 0.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 ratios: List[float] = [0.5, 1.0, 2.0],  # noqa: B006
                 fraction: float = 0.5,
                 fpn_layer: str = 'p3',
                 scales: Optional[List[float]] = None,
                 base_sizes: Optional[List[int]] = None,
                 scale_major: bool = True,
                 centers: Optional[List[Tuple[float, float]]] = None,
                 center_offset: float = 0.):
        # Check center and center_offset
        if center_offset != 0:
            assert centers is None, (
                'center cannot be set when center_offset != 0, '
                f'{centers} is given.')
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # Calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        if scales is None:
            scales = [1.0]
        self.scales = torch.Tensor(scales)

        self.fraction = fraction
        self.fpn_layer = fpn_layer
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator is applied to"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors using theoretical receptive field sizes.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple
                feature levels.
        """
        multi_level_base_anchors = []
        all_trfs = self.gen_trf()
        if self.fpn_layer == 'p3':
            # TRF of P3~P7 FPN for single-stage detectors
            self.base_sizes = all_trfs[-5:]
        else:
            # TRF of P2~P6 FPN for two-stage detectors
            self.base_sizes = all_trfs[:5]

        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=torch.tensor([1.0]),
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    @staticmethod
    def gen_trf():
        """Calculate the theoretical receptive field from P2-P7 of a
        standard ResNet-50-FPN.

        Reference: https://distill.pub/2019/computing-receptive-fields/

        Returns:
            list[int]: TRF sizes for [P2, P3, P4, P5, P6, P7].
        """
        j_i = [1]
        for i in range(7):
            j = j_i[i] * 2
            j_i.append(j)

        r0 = 1
        r1 = r0 + (7 - 1) * j_i[0]

        r2 = r1 + (3 - 1) * j_i[1]
        trf_p2 = r2 + (3 - 1) * j_i[2] * 3

        r3 = trf_p2 + (3 - 1) * j_i[2]
        trf_p3 = r3 + (3 - 1) * j_i[3] * 3

        r4 = trf_p3 + (3 - 1) * j_i[3]
        trf_p4 = r4 + (3 - 1) * j_i[4] * 5

        r5 = trf_p4 + (3 - 1) * j_i[4]
        trf_p5 = r5 + (3 - 1) * j_i[5] * 2

        trf_p6 = trf_p5 + (3 - 1) * j_i[6]

        trf_p7 = trf_p6 + (3 - 1) * j_i[7]

        trfs = [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]
        return trfs

    def gen_single_level_base_anchors(self,
                                      base_size: int,
                                      scales: Tensor,
                                      ratios: Tensor,
                                      center: Optional[tuple] = None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor (from TRF).
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between height and width
                of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                relative to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature map, shape
                (num_anchors, 4) in <x1, y1, x2, y2> format.
        """
        w = base_size * self.fraction
        h = base_size * self.fraction
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs,
            x_center + 0.5 * ws, y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    @staticmethod
    def _meshgrid(x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self,
                    featmap_sizes: List[tuple],
                    dtype: torch.dtype = torch.float32,
                    device: DeviceType = 'cuda') -> List[Tensor]:
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            dtype (torch.dtype): Data type of the anchors.
                Default torch.float32.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels.
                Each tensor has shape [N, 4], where
                N = width * height * num_base_anchors.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,
                                 featmap_size: tuple,
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: DeviceType = 'cuda') -> Tensor:
        """Generate grid anchors of a single level.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (torch.dtype): Data type. Default torch.float32.
            device (str): Device to put tensors on. Default 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps, shape
                (h * w * num_base_anchors, 4).
        """
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(
            0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(
            0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack(
            [shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self,
                    featmap_sizes: List[tuple],
                    pad_shape: tuple,
                    device: DeviceType = 'cuda') -> List[Tensor]:
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w),
                (valid_feat_h, valid_feat_w),
                self.num_base_anchors[i],
                device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    @staticmethod
    def single_level_valid_flags(featmap_size: tuple,
                                 valid_size: tuple,
                                 num_base_anchors: int,
                                 device: DeviceType = 'cuda') -> Tensor:
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str): Device. Default 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level
                feature map, shape (h * w * num_base_anchors,).
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = RFGenerator._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str
