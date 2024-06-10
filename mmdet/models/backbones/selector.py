import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class Selector_test1(BaseModule):
    '''test use'''
    def __init__(self, input_channels, num_experts, dropout=0.2,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                 ):
        super().__init__(init_cfg=init_cfg)
        hidden_dim = int(input_channels // 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
