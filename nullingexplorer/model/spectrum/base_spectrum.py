# 光谱基类，forward返回特定光谱的发光线形
import torch.nn as nn

from nullingexplorer.utils import ModuleRegister

class BaseSpectrum(ModuleRegister, cls_type='Spectrum', cls_name='base'):
    def __init_subclass__(cls, cls_name=None):
        return super().__init_subclass__(cls_type='Spectrum', cls_name=cls_name)

    def __init__(self):
        super(BaseSpectrum, self).__init__()

    def forward(self, x):
        pass