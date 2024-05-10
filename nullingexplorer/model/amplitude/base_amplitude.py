# 振幅基类，forward返回每个待观测目标进入阵列望远镜集光器的光子数
import torch.nn as nn

from nullingexplorer.utils import ModuleRegister

class BaseAmplitude(ModuleRegister, cls_type='Amplitude', cls_name='base'):
    def __init_subclass__(cls, cls_name=None):
        return super().__init_subclass__(cls_type='Amplitude', cls_name=cls_name)

    def __init__(self):
        super(BaseAmplitude, self).__init__()

    def forward(self, x):
        pass