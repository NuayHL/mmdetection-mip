import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

@MODELS.register_module()
class CV_Squared_Loss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def __call__(self, x):
        """x should be the full size of gate value"""
        x = x.sum(dim=0)
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor(0, device=x.device, dtype=x.dtype)
        return self.loss_weight * x.float().var() / (x.float().mean() ** 2 + eps)
