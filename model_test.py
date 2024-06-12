import torch
# import mmdet.models.data_preprocessors
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
register_all_modules()

cfg = Config.fromfile('configs/yolox/yolox_s_test1_mip_4xb16-300e_voc.py')

detector = MODELS.build(cfg.model).cuda()

input_tensor = torch.randn((3, 3, 640, 640)).cuda()

predict = detector(input_tensor, mode='predict')

print(predict)

