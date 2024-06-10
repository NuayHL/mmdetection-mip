import math
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class BackboneWithGate(BaseModule):
    def __init__(self,
                 backbone_cfg,
                 gate_cfg,
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               a=math.sqrt(5),
                               distribution='uniform',
                               mode='fan_in',
                               nonlinearity='leaky_relu')):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone_cfg)
        self.gate = MODELS.build(gate_cfg)

    def train(self, mode=True):
        super().train(mode)

    def forward(self, x):
        backbone_output = self.backbone(x)
        last_level = backbone_output[-1]
        gate_value = self.gate(last_level)
        return backbone_output, gate_value



