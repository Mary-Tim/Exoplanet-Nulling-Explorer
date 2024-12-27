# 光谱基类，forward返回特定光谱的发光线形
import torch.nn as nn

from nullingexplorer.utils import ModuleRegister

class BaseOrbit(ModuleRegister, cls_type='Orbit', cls_name='base'):
    def __init_subclass__(cls, cls_name=None):
        return super().__init_subclass__(cls_type='Orbit', cls_name=cls_name)

    def __init__(self, *args, **kwargs):
        super(BaseOrbit, self).__init__()

    def forward(self, x):
        pass