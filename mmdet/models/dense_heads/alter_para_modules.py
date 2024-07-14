import warnings

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class AlterConvModule(ConvModule):
    """ build for only alter the bias value of the nn.Conv2d """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.conv, nn.Conv2d)
        assert not self.with_explicit_padding

    def forward(self,
                x: torch.Tensor,
                bias: nn.Parameter,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == 'conv':
                x = nn.functional.conv2d(x, self.conv.weight, bias=bias, stride=self.stride,
                                         padding=self.padding)
                # x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x

@MODELS.register_module()
class AlterConv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=False,
                              padding_mode=padding_mode,
                              device=device,
                              dtype=dtype)
        self.stride = stride
        self.padding = padding

    def forward(self, x, bias):
        return nn.functional.conv2d(x, self.conv.weight, bias=bias, stride=self.stride, padding=self.padding)

@MODELS.register_module()
class AlterModuleSeq(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.model_seq = nn.ModuleList(modules)

    def __len__(self):
        return len(self.model_seq)

    def init_weights(self):
        for module in self.model_seq:
            if hasattr(module, 'init_weights'):
                module.init_weights()
            else:
                warnings.warn(f'The module {type(module)} has no \'init_weight\' func')

    def forward(self, x, add_pars: list[dict]):
        assert len(add_pars) == len(self)
        for model, add_par in zip(self.model_seq, add_pars):
            x = model(x, **add_par)
        return x

