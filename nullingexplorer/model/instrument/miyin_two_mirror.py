import torch
import torch.nn as nn

from .base_instrument import BaseInstrument
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

class MiYinTwoMirror(BaseInstrument):
    def __init__(self):
        super(MiYinTwoMirror, self).__init__()
        
        # Instrument constant parameters
        self.register_buffer('mirror_diameter', cfg.get_property('mirror_diameter'))      # Diameter of each mirror (unit: meter)
        self.register_buffer('quantum_eff', cfg.get_property('quantum_eff'))       # Quantum efficiency of detector (dimensionless)
        self.register_buffer('instrument_eff', cfg.get_property('instrument_eff'))   # Instrument throughput efficiency (dimensionless)
        self.register_buffer('fov_scale', cfg.get_property('fov_scale'))

        self.register_buffer('total_eff', 2. * torch.pi * (self.mirror_diameter/2.)**2 * self.quantum_eff * self.instrument_eff)

        # register nn.Tanh for FoV estimation
        self.tanh = nn.Tanh()

    def forward(self, data):
        '''
        Return the detect efficiency times the integral time

        : param data: dict of dataset, content:
            integral_time: Integral time of each data point (unit: second)
        '''
        return self.total_eff * data['intg_time']

    def field_of_view(self, ra, dec, wavelength):
        '''
        Calculate the effective field-of-view

        : param ra: right ascension of the point (unit: radius)
        : param dec: declination of the point (unit: radius)
        : param wavelength: wavelength of the light (unit: meter)
        '''
        return self.field_of_view_polar(torch.sqrt(ra**2+dec**2), wavelength)
        #return 0.5 * (1 - self.tanh(cons._radian_to_mas*(torch.sqrt(ra**2+dec**2)-0.5*wavelength/(self.mirror_diameter))))

    def field_of_view_polar(self, angular, wavelength):
        '''
        Calculate the effective field-of-view

        : param ra: right ascension of the point (unit: radius)
        : param dec: declination of the point (unit: radius)
        : param wavelength: wavelength of the light (unit: meter)
        '''
        return 0.5 * (1 - self.tanh(cons._radian_to_mas*(angular/self.fov_scale-0.5*wavelength/(self.mirror_diameter))))