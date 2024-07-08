import torch
import torch.nn as nn
import numpy as np
from nullingexplorer.model.transmission import BaseTransmission
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.utils import Constants as cons

class DualChoppedDestructive(BaseTransmission):
    def __init__(self):
        super().__init__()
        self.register_buffer('half_baseline', cfg.get_property('baseline') / 2.)
        self.register_buffer('ratio', cfg.get_property('ratio'))
        self.register_buffer('nulling_scale', 1. - 2.*cfg.get_property('nulling_depth'))

    def forward(self, ra, dec, wavelength, data):
        '''
        Transmission map of a dual-chopped Bracewell nuller

        : param ra: right ascension of the point (unit: radius)
        : param dec: declination of the point (unit: radius)
        : param data: dict of dataset, content:
            phi: phase of nuller (unitL radian)
            wavelength: wavelength of the light (unit: meter)
            mod: flag value, select output model 3 (+1) or 4 (-1) (unit: dimensionless)
        '''
        alpha, beta = self.cartesian_rotation(ra, dec, -data.phase)
        if hasattr(data, 'baseline'):
            trans_map = torch.sin(2 * np.pi * data.baseline / 2. * alpha / wavelength) ** 2 * torch.cos(
                        2 * self.ratio * np.pi * data.baseline / 2. * beta / wavelength - data.mod * torch.pi / 4.) ** 2
        else:
            trans_map = torch.sin(2 * np.pi * self.half_baseline * alpha / wavelength) ** 2 * torch.cos(
                        2 * self.ratio * np.pi * self.half_baseline * beta / wavelength - data.mod * torch.pi / 4.) ** 2
        return (trans_map - 0.5) * self.nulling_scale + 0.5

class DualChoppedDifferential(BaseTransmission):
    def __init__(self):
        super().__init__()
        self.register_buffer('half_baseline', cfg.get_property('baseline') / 2.)
        self.register_buffer('ratio', cfg.get_property('ratio'))

    def forward(self, ra, dec, wavelength, data):
        '''
        Transmission map of a dual-chopped Bracewell nuller

        : param ra: right ascension of the point (unit: radius)
        : param dec: declination of the point (unit: radius)
        : param data: dict of dataset, content:
            phi: phase of nuller (unitL radian)
            wavelength: wavelength of the light (unit: meter)
        '''
        alpha, beta = self.cartesian_rotation(ra, dec, -data.phase)
        if hasattr(data, 'baseline'):
            return torch.sin(2 * np.pi * data.baseline / 2. * alpha / wavelength) ** 2 * torch.sin(
                        4 * self.ratio * np.pi * data.baseline / 2. * beta / wavelength)
        else:
            return torch.sin(2 * np.pi * self.half_baseline * alpha / wavelength) ** 2 * torch.sin(
                        4 * self.ratio * np.pi * self.half_baseline * beta / wavelength)