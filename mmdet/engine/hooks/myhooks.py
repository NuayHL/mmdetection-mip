from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS

import math

@HOOKS.register_module()
class ChangeTempHook(Hook):
    def __init__(self, fin_temp=10.0):
        super().__init__()
        self.max_epoch = 300
        self.init_temp = 1.0
        self.fin_temp = fin_temp
        self.current_epoch = 0
        self.is_dist = False

    @property
    def increase_interval(self):
        return self.fin_temp - self.init_temp

    @property
    def step(self):
        return self.current_epoch

    def before_train(self, runner) -> None:
        self.max_epoch = runner.max_epochs
        model = runner.model
        if is_model_wrapper(runner.model):
            model = model.module
            self.is_dist = True
        assert hasattr(model, 'temperature'), (
            runner.logger.error(f'The module {runner.model_name}.bbox_head has no attribute \'temperature\'. '
                                f'Considering delete \'ChangeTempHook\' hook in config'))
        self.init_temp = model.bbox_head.temperature
        runner.logger.info(f'Apply dynamic temperature. Init temp={self.init_temp}, Fin temp={self.fin_temp}')

    def before_train_epoch(self, runner) -> None:
        self.current_epoch = runner.epoch
        _progress = 0.5 * (1 - math.cos((self.step + 1) * math.pi / self.max_epoch))
        if self.is_dist:
            runner.model.module.bbox_head.temperature = _progress * self.increase_interval + self.init_temp
        else:
            runner.model.bbox_head.temperature = _progress * self.increase_interval + self.init_temp
        runner.logger.info(f'Epoch[{self.step}/{self.max_epoch}], model temp change to {runner.model.bbox_head.temperature}')


