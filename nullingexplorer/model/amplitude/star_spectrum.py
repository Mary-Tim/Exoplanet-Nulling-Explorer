import torch
import torch.nn as nn
from torchquad import Boole

from .base_amplitude import BaseAmplitude
from nullingexplorer.model.spectrum import BlackBody
from nullingexplorer.model.transmission import DualChoppedDestructive, SingleBracewell
from nullingexplorer.utils import Constants, get_transmission
from nullingexplorer.utils import Configuration as cfg

class StarBlackBody(BaseAmplitude):
    def __init__(self):
        super(StarBlackBody, self).__init__()
        # Surface Spectrum of the Planet
        self.spectrum = BlackBody()
        #self.trans_map = SingleBracewell()
        #self.trans_map = DualChoppedDestructive()
        self.trans_map = get_transmission(cfg.get_property('trans_map'))()
        # Constant parameters
        self.register_buffer('distance', cfg.get_property('distance') * Constants._pc_to_meter) # Distance between target and format (unit: pc)
        self.register_buffer('radius', cfg.get_property('star_radius') * 1e3) # Star radius (unit: kilometer)
        self.register_buffer('temperature', cfg.get_property('star_temperature')) # Star radius (unit: kilometer)
        # Calculate stellar leak
        self.register_buffer('sun_light', torch.tensor([], dtype=torch.float64))  

    def init_sun_light(self, data) -> torch.Tensor:
        integrator = Boole()
        integrate_jit_compiled_parts = integrator.get_jit_compiled_integrate(
            dim=3, N=100000, backend="torch"
        )

        wl_lo = data.wavelength - data.wl_width / 2.
        wl_hi = data.wavelength + data.wl_width / 2.

        wl_quad_domains = torch.stack((wl_lo, wl_hi)).t()

        list_sun_light = []
        for point, domain in zip(data, wl_quad_domains):
            def sun_light(x):
                ra, dec = self.trans_map.to_cartesian(x[:,0] / self.distance, x[:,1])
                return self.spectrum(self.temperature, x[:,2]) * self.trans_map(ra, dec, x[:,2], point) * x[:,0]
            list_sun_light.append(integrate_jit_compiled_parts(sun_light, torch.tensor([[0., self.radius], [-torch.pi, torch.pi], domain])))

        self.sun_light.data = torch.stack(list_sun_light) / self.distance**2

    def forward(self, data):
        # return the number of photons generated by the planet
        if len(self.sun_light) == 0:
            self.init_sun_light(data)

        return self.sun_light

        # TODO: Consider the field-of-view of instrument, using Sigmod?

    def star_luminosity(self):
        return torch.tensor(self.radius**2 * (self.temperature / 5780.)**4)

class StarBlackBodyFast(BaseAmplitude):
    def __init__(self):
        super().__init__()
        # Surface Spectrum of the Planet
        self.spectrum = BlackBody()
        self.trans_map = get_transmission(cfg.get_property('trans_map'))()
        # Constant parameters
        self.register_buffer('distance', cfg.get_property('distance') * Constants._pc_to_meter) # Distance between target and format (unit: pc)
        self.register_buffer('radius', cfg.get_property('star_radius') * 1e3) # Star radius (unit: kilometer)
        self.register_buffer('temperature', cfg.get_property('star_temperature')) # Star radius (unit: kilometer)
        # Calculate stellar leak
        self.register_buffer('sun_light', torch.tensor([], dtype=torch.float64))  

    def init_sun_light(self, data) -> torch.Tensor:
        phase_num = data.get_bin_number('phase')
        data_0 = data.select_data('phase', data.get_bins('phase')[0])
        integrator = Boole()
        integrate_jit_compiled_parts = integrator.get_jit_compiled_integrate(
            dim=3, N=100000, backend="torch"
        )

        wl_lo = data_0.wavelength - data_0.wl_width / 2.
        wl_hi = data_0.wavelength + data_0.wl_width / 2.

        wl_quad_domains = torch.stack((wl_lo, wl_hi)).t()

        list_sun_light = []
        for point, domain in zip(data_0, wl_quad_domains):
            def sun_light(x):
                ra, dec = self.trans_map.to_cartesian(x[:,0] / self.distance, x[:,1])
                return self.spectrum(self.temperature, x[:,2]) * self.trans_map(ra, dec, x[:,2], point) * x[:,0]
            list_sun_light.append(integrate_jit_compiled_parts(sun_light, torch.tensor([[0., self.radius], [-torch.pi, torch.pi], domain])))

        self.sun_light.data = torch.stack(list_sun_light).repeat(phase_num) / self.distance**2

    def forward(self, data):
        # return the number of photons generated by the planet
        if len(self.sun_light) == 0:
            self.init_sun_light(data)

        return self.sun_light

        # TODO: Consider the field-of-view of instrument, using Sigmod?

    def star_luminosity(self):
        return torch.tensor(self.radius**2 * (self.temperature / 5780.)**4)