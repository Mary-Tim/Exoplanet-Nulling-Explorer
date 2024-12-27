# 仪器基类，forward返回仪器对每个光谱Bin的探测效率
import torch.nn as nn

from nullingexplorer.utils import ModuleRegister

class BaseInstrument(ModuleRegister, cls_type='Instrument', cls_name='base'):
    def __init_subclass__(cls, cls_name=None):
        return super().__init_subclass__(cls_type='Instrument', cls_name=cls_name)

    def __init__(self):
        super(BaseInstrument, self).__init__()

    def forward(self, x):
        pass