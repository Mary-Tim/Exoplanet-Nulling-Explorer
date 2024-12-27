# 传输图基类，返回在指定基线b，阵列相位\phi与成像基线-消零基线比例r时，天区特定位置（ra, dec）每个光谱Bin对应的传输图效率
import torch
import torch.nn as nn

from nullingexplorer.utils import ModuleRegister

class BaseTransmission(ModuleRegister, cls_type='Transmission', cls_name='base'):
    def __init_subclass__(cls, cls_name=None):
        return super().__init_subclass__(cls_type='Transmission', cls_name=cls_name)

    def __init__(self, is_planet=False):
        super(BaseTransmission, self).__init__()
        self.is_planet = is_planet

    def forward(self, x):
        pass

    def to_polar(self, x, y):
        return torch.sqrt((x**2 + y**2)), torch.atan2(y, x)

    def to_cartesian(self, radius, theta):
        return radius * torch.cos(theta), radius * torch.sin(theta)

    def cartesian_rotation(self, x, y, phi):
        radius, theta = self.to_polar(x, y)
        theta = theta + phi
        return self.to_cartesian(radius, theta)