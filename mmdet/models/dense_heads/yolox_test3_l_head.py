# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops.nms import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, reduce_mean, list_slice, dict_sum_up)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..task_modules.samplers import PseudoSampler
from ..utils import multi_apply
from .base_dense_head import BaseDenseHead

from .alter_para_modules import AlterConvModule, AlterConv2D, AlterModuleSeq


class _YOLOX_test3_head_Experts(nn.Module):
    def __init__(self, strides, feat_channels, num_classes):
        super().__init__()
        self.strides = strides
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.multi_level_cls_convs = nn.ParameterList()
        self.multi_level_reg_convs = nn.ParameterList()
        self.multi_level_conv_cls = nn.ParameterList()
        self.multi_level_conv_reg = nn.ParameterList()
        self.multi_level_conv_obj = nn.ParameterList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(nn.Parameter(torch.zeros((2, self.feat_channels))))
            self.multi_level_reg_convs.append(nn.Parameter(torch.zeros((2, self.feat_channels))))
            self.multi_level_conv_cls.append(nn.Parameter(torch.zeros(self.num_classes)))
            self.multi_level_conv_reg.append(nn.Parameter(torch.zeros(4)))
            self.multi_level_conv_obj.append(nn.Parameter(torch.zeros(1)))
    
    def get_cls_convs(self):
        return [[{'bias': para[0]}, {'bias': para[1]}] for para in self.multi_level_cls_convs]

    def get_reg_convs(self):
        return [[{'bias': para[0]}, {'bias': para[1]}] for para in self.multi_level_reg_convs]

    def get_conv_cls(self):
        return self.multi_level_conv_cls

    def get_conv_reg(self):
        return self.multi_level_conv_reg

    def get_conv_obj(self):
        return self.multi_level_conv_obj
    

@MODELS.register_module()
class YOLOX_test3_l_Head(BaseDenseHead):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Defaults to 256
        stacked_convs (int): Number of stacking convs of the head.
            Defaults to (8, 16, 32).
        strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to None.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Defaults to False.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        loss_l1 (:obj:`ConfigDict` or dict): Config of L1 loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.

    My modification: light weight experts only changes the bias at head

    """

    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            feat_channels: int = 256,
            stacked_convs: int = 2,
            strides: Sequence[int] = (8, 16, 32),
            num_experts: int = 5,
            num_selects: int = 3,
            use_depthwise: bool = False,
            dcn_on_last_conv: bool = False,
            conv_bias: Union[bool, str] = 'auto',
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='Swish'),
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            loss_bbox: ConfigType = dict(
                type='IoULoss',
                mode='square',
                eps=1e-16,
                reduction='sum',
                loss_weight=5.0),
            loss_obj: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            loss_l1: ConfigType = dict(
                type='L1Loss', reduction='sum', loss_weight=1.0),
            loss_gate: ConfigType = dict(
                type='CV_Squared_Loss', loss_weight=0.1),
            loss_gate_obj_loss: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                reduction='mean',
                loss_weight=1.0),
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
    ) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        self.loss_obj: nn.Module = MODELS.build(loss_obj)
        self.loss_gate: nn.Module = MODELS.build(loss_gate)
        self.loss_gate_obj_loss:nn.Module = MODELS.build(loss_gate_obj_loss)

        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_l1: nn.Module = MODELS.build(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            # YOLOX does not support sampling
            self.sampler = PseudoSampler()

        self.softmax = nn.Softmax(dim=1)
        self._init_all()

    def _init_all(self):
        """Initialize the experts according to the num of experts"""
        self._build_shared_head()
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            self.experts.append(_YOLOX_test3_head_Experts(strides=self.strides, feat_channels=self.feat_channels,
                                                          num_classes=self.num_classes))

    def _build_shared_head(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        assert not self.use_depthwise
        assert not self.dcn_on_last_conv
        conv = AlterConvModule
        stacked_convs = list()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return AlterModuleSeq(*stacked_convs)

    def _build_predictor(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_cls = AlterConv2D(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = AlterConv2D(self.feat_channels, 4, 1)
        conv_obj = AlterConv2D(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super(YOLOX_test3_l_Head, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for expert in self.experts:
            for conv_cls, conv_obj in zip(expert.multi_level_conv_cls,
                                          expert.multi_level_conv_obj):
                conv_cls.data.fill_(bias_init)
                conv_obj.data.fill_(bias_init)

    def forward_single(self, x: Tensor,
                       cls_convs: nn.Module, cls_convs_bias: list[dict],
                       reg_convs: nn.Module, reg_convs_bias: list[dict],
                       conv_cls: nn.Module, conv_cls_bias: nn.Parameter,
                       conv_reg: nn.Module, conv_reg_bias: nn.Parameter,
                       conv_obj: nn.Module, conv_obj_bias: nn.Parameter) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x, cls_convs_bias)
        reg_feat = reg_convs(x, reg_convs_bias)

        cls_score = conv_cls(cls_feat, conv_cls_bias)
        bbox_pred = conv_reg(reg_feat, conv_reg_bias)
        objectness = conv_obj(reg_feat, conv_obj_bias)
        return cls_score, bbox_pred, objectness

    def forward(self, x: Tuple[Tensor]):
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple(Tuple[List], gate_value):
            (
            A tuple of multi-level classification scores,
                                   bbox predictions,
                                   and objectnesses. ,
            The return of self.gate_value_to_topk(gate values)
            )
        """
        gate_value = x[-1]
        x = x[:-1]

        gate_value, topk_idx = self.gate_value_to_topk(gate_value)
        batch_size = topk_idx.size(0)
        distribute_idx = self.distribute_to_experts(topk_idx=topk_idx)

        raw_result = list()

        for i in range(self.num_experts):
            task_idx = distribute_idx[i]
            if task_idx is None:
                if self.training:
                    # move the random pick here
                    task_idx = torch.randint(0, batch_size, (1,), device=topk_idx.device)
                    distribute_idx[i] = task_idx
                else:
                    raw_result.append(None)
                    continue

            i_x = tuple([_x[task_idx] for _x in x])  # slice the input

            raw_result.append(multi_apply(self.forward_single, i_x,
                                          self.multi_level_cls_convs, self.experts[i].get_cls_convs(),
                                          self.multi_level_reg_convs, self.experts[i].get_reg_convs(),
                                          self.multi_level_conv_cls,  self.experts[i].get_conv_cls(),
                                          self.multi_level_conv_reg,  self.experts[i].get_conv_reg(),
                                          self.multi_level_conv_obj,  self.experts[i].get_conv_obj()))

        return raw_result, (gate_value, topk_idx, distribute_idx)

    def predict_by_feat(self,
                        experts_output_list,
                        gate_value_and_topk_idx,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> List[InstanceData]:
        """Doing nms after gathering"""
        # always None
        cfg = self.test_cfg if cfg is None else cfg

        gate_value, topk_idx, distribute_idx = gate_value_and_topk_idx
        batch_lenth = len(batch_img_metas)
        result_list = [None] * batch_lenth
        for i in range(self.num_experts):
            task_idx = distribute_idx[i]
            # if no data sent to this expert
            if task_idx is None:
                continue
            cls_scores, bbox_preds, objectnesses = experts_output_list[i]
            batch_img_metas_i = list_slice(task_idx, batch_img_metas)

            predicts_list = self._predict_by_feat(cls_scores,
                                                  bbox_preds,
                                                  objectnesses,
                                                  batch_img_metas_i,
                                                  cfg=cfg,
                                                  rescale=rescale,
                                                  with_nms=with_nms)
            for idx, batch_idx in enumerate(task_idx.tolist()):
                # ensuring that the instance is not None even if empty
                if result_list[batch_idx] is None:
                    result_list[batch_idx] = predicts_list[idx]
                else:
                    if len(predicts_list[idx]) == 0:
                        continue
                    else:
                        result_list[batch_idx] = result_list[batch_idx].cat(list(predicts_list[idx]))

        # Doing nms here
        for idx, img_meta in enumerate(batch_img_metas):
            result_list[idx] = self._bbox_post_process(results=result_list[idx],
                                                       cfg=cfg,
                                                       rescale=rescale,
                                                       with_nms=with_nms,
                                                       img_meta=img_meta)
        return result_list

    def _predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        result_list = []
        for img_id in range(num_imgs):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = flatten_objectness[
                             img_id] * max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=flatten_bboxes[img_id][valid_mask],
                scores=max_scores[valid_mask] *
                       flatten_objectness[img_id][valid_mask],
                labels=labels[valid_mask])

            # no nms here
            result_list.append(results)

        return result_list

    def _bbox_decode(self, priors: Tensor, bbox_preds: Tensor) -> Tensor:
        """Decode regression results (delta_x, delta_x, w, h) to bboxes (tl_x,
        tl_y, br_x, br_y).

        Args:
            priors (Tensor): Center proiors of an image, has shape
                (num_instances, 2).
            bbox_preds (Tensor): Box energies / deltas for all instances,
                has shape (batch_size, num_instances, 4).

        Returns:
            Tensor: Decoded bboxes in (tl_x, tl_y, br_x, br_y) format. Has
            shape (batch_size, num_instances, 4).
        """
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        if rescale:
            assert img_meta.get('scale_factor') is not None
            results.bboxes /= results.bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results

    def loss_by_feat(self,
                     experts_output_list,
                     gate_value_and_topk_idx,
                     batch_gt_instances: Sequence[InstanceData],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        gate_value, topk_idx, distribute_idx = gate_value_and_topk_idx

        selected_obj_loss = torch.zeros_like(gate_value, requires_grad=False).type_as(gate_value)
        selected_idx, _ = torch.sort(topk_idx, dim=1)

        loss_dict = dict()
        for i in range(self.num_experts):
            task_idx = distribute_idx[i]
            cls_scores, bbox_preds, objectnesses = experts_output_list[i]
            # random pick futher needs to be improved
            if task_idx is None:
                task_idx = torch.randint(0, len(batch_img_metas), (1,), device=topk_idx.device)
            batch_gt_instances_i = list_slice(task_idx, batch_gt_instances)
            batch_img_metas_i = list_slice(task_idx, batch_img_metas)
            batch_gt_instances_ignore_i = list_slice(task_idx, batch_gt_instances_ignore)
            _loss_dict, _obj_loss_label = self._loss_by_feat(cls_scores,
                                                             bbox_preds,
                                                             objectnesses,
                                                             batch_gt_instances_i,
                                                             batch_img_metas_i,
                                                             batch_gt_instances_ignore_i)
            selected_obj_loss[:, i] = selected_obj_loss[:, i].scatter(0, task_idx, _obj_loss_label)
            loss_dict = dict_sum_up(loss_dict, _loss_dict)
        batch_indices = torch.arange(selected_obj_loss.size(0), device=selected_idx.device).unsqueeze(1)
        selected_obj_loss = selected_obj_loss[batch_indices, selected_idx]
        selected_obj_loss = torch.softmax(selected_obj_loss, dim=1)
        selected_fin_obj_loss = torch.zeros_like(gate_value, requires_grad=False).type_as(gate_value)
        selected_fin_obj_loss = selected_fin_obj_loss.scatter(1, selected_idx, selected_obj_loss)
        loss_dict['loss_gate_obj'] = self.loss_gate_obj_loss(gate_value, selected_fin_obj_loss)

        loss_dict['loss_gate'] = self.loss_gate(gate_value)

        return loss_dict

    def gate_value_to_topk(self, gate_value):
        """how to use the gate value, can be implemented with various method
            e.g. noise top-k
            current is the easiest implement

            input: torch shape (batch_size, num_experts)
            return:
                fin_gate_value: for load balance loss
                topk_idx: for distribution to each experts
        """
        top_logists, topk_idx = gate_value.topk(self.num_selects, dim=1)
        top_gate_value = self.softmax(top_logists)

        """set the rest to zero, while keep the gate value shape"""
        fin_gate_value = torch.zeros_like(gate_value, requires_grad=True).type_as(top_gate_value)
        fin_gate_value = fin_gate_value.scatter(1, topk_idx, top_gate_value)

        return fin_gate_value, topk_idx

    def distribute_to_experts(self, topk_idx):
        """distribute task within a batch to different experts
            input topk_idx: shape (batch_size, num_selects）
        """
        output = [None] * self.num_experts
        for expert_id in range(self.num_experts):
            data_indices = (topk_idx == expert_id).nonzero(as_tuple=True)[0]
            if data_indices.numel() > 0:
                output[expert_id] = data_indices
        return output

    def _loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None):
        """Compute the loss of each expert,
            Add the distribute_batch_idx to calculate certain idx.

            Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
            tensor[num_image]: obj_loss of each assigned data
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_targets_single,
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_cls_preds.detach(), flatten_bboxes.detach(),
             flatten_objectness.detach(), batch_gt_instances, batch_img_metas,
             batch_gt_instances_ignore)

        # The experimental results show that 'reduce_mean' can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)        # list( (8400) )
        cls_targets = torch.cat(cls_targets, 0)    # list( (pos_num, 20) )
        # obj_targets = torch.cat(obj_targets, 0)    # list( (8400, 1) )
        obj_targets = torch.cat([obj_target.permute(1, 0) for obj_target in obj_targets], 0)
        bbox_targets = torch.cat(bbox_targets, 0)  # list( (pos_num, 4))

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)  # list( (pos_num, 4))

        loss_obj = torch.sum(self.loss_obj(flatten_objectness, obj_targets, reduction_override='none'), dim=1)
        loss_obj_label = loss_obj.detach()  # for gate output label
        loss_obj = torch.sum(loss_obj) / num_total_samples

        # loss_obj = self.loss_obj(flatten_objectness, obj_targets, reduction_override='none') / num_total_samples

        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                cls_targets) / num_total_samples
            loss_bbox = self.loss_bbox(
                flatten_bboxes.view(-1, 4)[pos_masks],
                bbox_targets) / num_total_samples
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            if num_pos > 0:
                loss_l1 = self.loss_l1(
                    flatten_bbox_preds.view(-1, 4)[pos_masks],
                    l1_targets) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_l1 = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict, loss_obj_label

    @torch.no_grad()
    def _get_targets_single(
            self,
            priors: Tensor,
            cls_preds: Tensor,
            decoded_bboxes: Tensor,
            objectness: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            gt_instances_ignore: Optional[InstanceData] = None) -> tuple:
        """Compute classification, regression, and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            tuple:
                foreground_mask (list[Tensor]): Binary mask of foreground
                targets.
                cls_target (list[Tensor]): Classification targets of an image.
                obj_target (list[Tensor]): Objectness targets of an image.
                bbox_target (list[Tensor]): BBox targets of an image.
                l1_target (int): BBox L1 targets of an image.
                num_pos_per_img (int): Number of positive samples in an image.
        """

        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)

    def _get_l1_target(self,
                       l1_target: Tensor,
                       gt_bboxes: Tensor,
                       priors: Tensor,
                       eps: float = 1e-8) -> Tensor:
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
