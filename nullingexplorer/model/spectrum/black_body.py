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
        #if self.c.get_device() != 0 or wavelength.get_device() != 0:
        #    print(f"Device: c: {self.c.get_device()}, wavelength: {wavelength.get_device()}")
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
        self.black_body = BlackBody()
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
        return self.black_body(temperature, data['wl_mid']) * (data['wl_hi'] - data['wl_lo'])

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
    '''
    TODO: Bug report:
        Traceback (most recent call last):
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/scan_signi_wavelength.py", line 132, in <module>
            significance[i] = sig_point(wl_lo, wl_hi)
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/scan_signi_wavelength.py", line 125, in sig_point
            sig_pe = sig_poisson.gen_sig_pe()
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/../../nullingexplorer/significance/poisson_significance.py", line 61, in gen_sig_pe
            sig_data['photon_electron'] = sig_amp(sig_data)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
            return forward_call(*args, **kwargs)
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/../../nullingexplorer/generator/amplitude_creator.py", line 70, in forward
            return torch.sum(torch.stack([getattr(self, name)(data) for name in self.config['Amplitude']]), 0) * self.instrument(data)
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/../../nullingexplorer/generator/amplitude_creator.py", line 70, in <listcomp>
            return torch.sum(torch.stack([getattr(self, name)(data) for name in self.config['Amplitude']]), 0) * self.instrument(data)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
            return forward_call(*args, **kwargs)
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/../../nullingexplorer/model/amplitude/planet_spectrum.py", line 37, in forward
            return torch.pi * (self.radius / (self.distance))**2 * self.spectrum(self.temperature, data) * \
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
            return self._call_impl(*args, **kwargs)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
            return forward_call(*args, **kwargs)
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/../../nullingexplorer/model/spectrum/black_body.py", line 132, in forward
            return torch.vmap(interp_black_body)(data, self.wl_interp)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/_functorch/apis.py", line 188, in wrapped
            return vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/_functorch/vmap.py", line 281, in vmap_impl
            return _flat_vmap(
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/_functorch/vmap.py", line 47, in fn
            return f(*args, **kwargs)
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/_functorch/vmap.py", line 403, in _flat_vmap
            batched_outputs = func(*batched_inputs, **kwargs)
          File "/home/mart/WorkSpace/Nulling/NullingExplorer/test/significance/../../nullingexplorer/model/spectrum/black_body.py", line 130, in <lambda>
            interp_black_body = lambda point, wl: torch.sum(self.unbinned_black_body(temperature, wl)) / self.interp_num * (point['wl_hi'] - point['wl_lo'])
          File "/home/mart/miniconda3/envs/NullExoExp/lib/python3.10/site-packages/torch/utils/_device.py", line 78, in __torch_function__
            return func(*args, **kwargs)
        IndexError: select(): index 0 out of range for tensor of size [0] at dimension 0
    '''
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