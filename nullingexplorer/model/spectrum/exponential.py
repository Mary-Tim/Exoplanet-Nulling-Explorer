import torch
import torch.nn as nn
from torchquad import Boole

from nullingexplorer.model.spectrum import BaseSpectrum
from nullingexplorer.utils import Constants

class Exponential(nn.Module):
    def __init__(self):
        super().__init__()
        # register constants
        self.register_buffer('c', torch.tensor(Constants._light_speed))
        self.register_buffer('exp_para', torch.tensor(Constants._Planck_constant * Constants._light_speed / Constants._Boltzmann_constant))

    def forward(self, temperature, wavelength):
        '''
        Planck law. 
        Density of the number of emmision photons by the black body. (unit: meter^{-1})

        : param temperature: <Tensor> Temperature of black body (unit: Kelvin)
        : param wavelength: <Tensor> light wavelength (unit: meter)
        '''
        return 2 * self.c / (torch.exp(self.exp_para / (wavelength * temperature)) -1) / torch.pow(wavelength, 4)