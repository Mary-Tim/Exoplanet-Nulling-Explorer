import torch
import torch.nn as nn
import numpy as np
from nullingexplorer.model.transmission import BaseTransmission
from nullingexplorer.utils import Configuration as cfg

class SingleBracewell(BaseTransmission):
    def __init__(self):
        super(SingleBracewell, self).__init__()
        self.register_buffer('baseline', cfg.get_property('baseline'))

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
        return 2 * torch.sin(torch.pi * radius * self.baseline / wavelength * torch.cos(theta - data.phi)) ** 2