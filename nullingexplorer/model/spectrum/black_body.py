import torch
import torch.nn as nn
from torchquad import Boole

from nullingexplorer.model.spectrum import BaseSpectrum
from nullingexplorer.utils import Constants

class BlackBody(nn.Module):
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


class UnbinnedBlackBody(BaseSpectrum):
    def __init__(self):
        super().__init__()
        # register constants
        self.register_buffer('c', torch.tensor(Constants._light_speed))
        self.register_buffer('exp_para', torch.tensor(Constants._Planck_constant * Constants._light_speed / Constants._Boltzmann_constant))
        self.black_body = BlackBody()

        # register parameters
        #self.temperature = nn.Parameter(torch.tensor(273.))  # Temperature of object (unit: Kelvin)

    def forward(self, temperature, data):
        '''
        Planck law. 
        Density of the number of emmision photons by the black body. (unit: meter^{-1})

        : param temperature: <Tensor> Temperature of black body (unit: Kelvin)
        : param wavelength: <Tensor> light wavelength (unit: meter)
        '''
        return self.black_body(temperature, data.wavelength)
        #return 2 * self.c / torch.pow(wavelength, 4) / (torch.exp(self.exp_para / (wavelength * temperature)) -1)

class BinnedBlackBody(BaseSpectrum):
    def __init__(self):
        super().__init__()
        # module
        self.unbinned_black_body = BlackBody()
        # register constants
        self.register_buffer('c', torch.tensor(Constants._light_speed))
        self.register_buffer('exp_para', torch.tensor(Constants._Planck_constant * Constants._light_speed / Constants._Boltzmann_constant))
        # register parameters
        #self.temperature = nn.Parameter(torch.tensor(273.))  # Temperature of object (unit: Kelvin)

    def forward(self, temperature, data):
        '''
        Planck law. 
        Number of emmision photons by the black body in each wavelength bin. (unit: meter^{-1})

        : param data: dict of dataset, content:
            wavelength: <Tensor> Center value of each wavelength bin (unit: meter)
            wl_width: <Tensor> Width of each wavelength bin (unit: meter)
        '''
        return self.unbinned_black_body(temperature, data.wavelength) * data.wl_width

class TorchQuadBlackBody(BaseSpectrum):
    def __init__(self):
        super().__init__()
        # module
        self.unbinned_black_body = BlackBody()
        # register constants
        self.register_buffer('c', torch.tensor(Constants._light_speed))
        self.register_buffer('exp_para', torch.tensor(Constants._Planck_constant * Constants._light_speed / Constants._Boltzmann_constant))
        # register parameters
        #self.temperature = nn.Parameter(torch.tensor(273.))  # Temperature of object (unit: Kelvin)
        integrator = Boole()
        self._integrate_jit_compiled_parts = integrator.get_jit_compiled_integrate(
            dim=1, backend="torch"
        )

    def forward(self, temperature, data):
        '''
        Planck law. 
        Number of emmision photons by the black body in each wavelength bin. (unit: meter^{-1})
        Evaluated by torchquad.

        : param data: torchclass of dataset, content:
            wavelength: <Tensor> Center value of each wavelength bin (unit: meter)
            wl_width: <Tensor> Width of each wavelength bin (unit: meter)
        '''
        wl_lo = data.wavelength - data.wl_width / 2.
        wl_hi = data.wavelength + data.wl_width / 2.

        quad_domains = torch.stack((wl_lo, wl_hi)).t().unsqueeze(1)
        #quad_domains.requires_grad = True

        return torch.stack([self._integrate_jit_compiled_parts(lambda w: self.unbinned_black_body(temperature, w), domain) for domain in quad_domains])