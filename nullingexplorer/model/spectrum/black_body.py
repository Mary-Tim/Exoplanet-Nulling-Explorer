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
        Density of the number of emission photons by the black body. (unit: meter^{-1})

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
        Density of the number of emission photons by the black body. (unit: meter^{-1})

        : param temperature: <Tensor> Temperature of black body (unit: Kelvin)
        : param wavelength: <Tensor> light wavelength (unit: meter)
        '''
        return self.black_body(temperature, data['wl_mid'])
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
        Number of emission photons by the black body in each wavelength bin. (unit: meter^{-1})

        : param data: dict of dataset, content:
            wavelength: <Tensor> Center value of each wavelength bin (unit: meter)
            wl_width: <Tensor> Width of each wavelength bin (unit: meter)
        '''
        return self.unbinned_black_body(temperature, data['wl_mid']) * (data['wl_hi'] - data['wl_lo'])

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
        Number of emission photons by the black body in each wavelength bin. (unit: meter^{-1})
        Evaluated by torchquad.

        : param data: tensorclass of dataset, content:
            wavelength: <Tensor> Center value of each wavelength bin (unit: meter)
            wl_width: <Tensor> Width of each wavelength bin (unit: meter)
        '''
        quad_domains = torch.stack((data['wl_lo'], data['wl_hi'])).t().unsqueeze(1)
        #quad_domains.requires_grad = True

        return torch.stack([self._integrate_jit_compiled_parts(lambda w: self.unbinned_black_body(temperature, w), domain) for domain in quad_domains])

class InterpBlackBody(BaseSpectrum):
    def __init__(self):
        super().__init__()
        # module
        self.unbinned_black_body = BlackBody()
        # register constants
        self.register_buffer('c', torch.tensor(Constants._light_speed))
        self.register_buffer('exp_para', torch.tensor(Constants._Planck_constant * Constants._light_speed / Constants._Boltzmann_constant))
        self.register_buffer('interp_num', torch.tensor(100))
        self.register_buffer('wl_interp', torch.tensor([]))

    def forward(self, temperature, data):
        '''
        Planck law. 
        Number of emission photons by the black body in each wavelength bin. (unit: meter^{-1})
        Evaluated by torchquad.

        : param data: tensorclass of dataset, content:
            wavelength: <Tensor> Center value of each wavelength bin (unit: meter)
            wl_width: <Tensor> Width of each wavelength bin (unit: meter)
        '''
        if len(self.wl_interp) == 0:
            def wl_interp_gen(wl_lo, wl_hi):
                return torch.linspace(wl_lo, wl_hi, self.interp_num)
            self.wl_interp = torch.vmap(wl_interp_gen)(data['wl_lo'], data['wl_hi'])

        #def interp_black_body(point, wl):
        #    return torch.sum(self.unbinned_black_body(temperature, wl)) / self.interp_num * (point['wl_hi'] - point['wl_lo'])
        interp_black_body = lambda point, wl: torch.sum(self.unbinned_black_body(temperature, wl)) / self.interp_num * (point['wl_hi'] - point['wl_lo'])

        return torch.vmap(interp_black_body)(data, self.wl_interp)
        #return torch.stack([self._integrate_jit_compiled_parts(lambda w: self.unbinned_black_body(temperature, w), domain) for domain in quad_domains])