import torch
import torch.nn as nn
import numpy as np
from nullingexplorer.model.transmission import BaseTransmission
from nullingexplorer.utils import Configuration as cfg

class UnifiedTransmission(BaseTransmission):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.))

    def forward(self, ra, dec, wavelength, data):
        '''
        Transmission map of a single Bracewell nuller

        : param ra: right ascension of the point (unit: radius)
        : param dec: declination of the point (unit: radius)
        : param data: dict of dataset, content:
            phi: phase of nuller (unitL radian)
            wavelength: wavelength of the light (unit: meter)
        '''
        return self.scale