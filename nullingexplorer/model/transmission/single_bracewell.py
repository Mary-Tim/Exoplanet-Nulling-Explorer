import torch
import torch.nn as nn
import numpy as np
from nullingexplorer.model.transmission import BaseTransmission
from nullingexplorer.utils import Configuration as cfg

class SingleBracewell(BaseTransmission):
    def __init__(self):
        super(SingleBracewell, self).__init__()
        #self.register_buffer('baseline', cfg.get_property('baseline'))
        self.register_buffer('nulling_scale', 1. - 2.*cfg.get_property('nulling_depth'))
        self.register_buffer('wl_interp_num', torch.tensor(100, dtype=torch.int))

        self._wl_interp = None

    def forward(self, ra, dec, wavelength, data):
        '''
        Transmission map of a single Bracewell nuller

        : param ra: right ascension of the point (unit: radius)
        : param dec: declination of the point (unit: radius)
        : param data: dict of dataset, content:
            phi: phase of nuller (unitL radian)
            wavelength: wavelength of the light (unit: meter)
        '''
        radius, theta = self.to_polar(ra, dec)
        trans_map = torch.sin(torch.pi * radius * data['baseline'] / wavelength * torch.cos(theta - data['phase'])) ** 2
        #if wavelength != -1:
        #    trans_map = torch.sin(torch.pi * radius * data['baseline'] / wavelength * torch.cos(theta - data['phase'])) ** 2
        #else:
        #    if self._wl_interp is None:
        #        self._wl_interp = torch.vmap(torch.linspace, in_dims=(0, 0, None))(data['wl_lo'], data['wl_hi'], self.wl_interp_num)
        #    trans_interp = torch.sin(torch.pi * radius * data['baseline'] / self._wl_interp * torch.cos(theta - data['phase'])) ** 2
        #    trans_map = torch.mean(trans_interp, dim=1)

        return (trans_map - 0.5) * self.nulling_scale + 0.5
        #return 2 * torch.sin(torch.pi * radius * data['baseline'] / wavelength * torch.cos(theta - data['phase'])) ** 2