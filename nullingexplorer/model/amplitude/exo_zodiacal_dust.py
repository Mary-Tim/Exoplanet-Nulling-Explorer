import torch
import torch.nn as nn
from torchquad import Boole, MonteCarlo, Trapezoid

from .base_amplitude import BaseAmplitude
from nullingexplorer.model.spectrum import BlackBody
from nullingexplorer.model.transmission import DualChoppedDestructive, SingleBracewell
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.utils import Constants, get_transmission
from nullingexplorer.utils import Configuration as cfg

class ExoZodiacalDust(BaseAmplitude):
    '''
    The photon emission by exo-zodiacal dust
    Reference to Kennedy 2015 (doi:10.1088/0067-0049/216/2/23)
    '''
    def __init__(self):
        super(ExoZodiacalDust, self).__init__()
        # Register models
        self.spectrum = BlackBody()
        #self.trans_map = SingleBracewell()
        #self.trans_map = DualChoppedDestructive()
        self.trans_map = get_transmission(cfg.get_property('trans_map'))()
        self.instrument = MiYinBasicType()
        # Constant parameters
        self.register_buffer('distance', cfg.get_property('distance') * Constants._pc_to_meter) # Distance between target and format (unit: pc)
        self.register_buffer('star_radius',   cfg.get_property('star_radius') * 1e3) # Star radius (unit: kilometer)
        self.register_buffer('star_temperature',   cfg.get_property('star_temperature')) # Star radius (unit: kilometer)
        self.register_buffer('zodi_level', cfg.get_property('zodi_level'))

        self.register_buffer('star_luminosity', (self.star_radius / 6.955e8)**2 * (self.star_temperature / 5772.)**4)
        self.register_buffer('alpha', torch.tensor(0.34))
        self.register_buffer('r_0', torch.sqrt(self.star_luminosity))
        self.register_buffer('r_in', 0.034422617777777775 * self.r_0)
        self.register_buffer('r_out', 10. * self.r_0)
        self.register_buffer('sigma_zero', torch.tensor(7.11889e-8))

        # Calculate exo-zodi
        self.register_buffer('exo_zodi', torch.tensor([], dtype=torch.float64))  

    def temperature(self, radium_au):
        #return 278.3 / torch.sqrt(radium_au)
        return 278.3 * (self.star_luminosity**0.25) / torch.sqrt(radium_au)

    def sigma_m(self, radium_au):
        return self.zodi_level * self.sigma_zero * (radium_au / self.r_0) ** (-self.alpha)

    def init_exo_zodi(self, data) -> torch.Tensor:

        wl_quad_domains = torch.stack((data['wl_lo'], data['wl_hi'])).t()

        integrator = Boole()
        integrate_jit_compiled_parts = integrator.get_jit_compiled_integrate(
            dim=3, N=300000, backend="torch"
        )
        list_exo_zodi = torch.zeros(len(data))
        for i, point in enumerate(data):
            def exo_zodi(x):
                theta = x[:,0] / self.distance
                ra, dec = self.trans_map.to_cartesian(theta, x[:,1])
                return self.spectrum(self.temperature(x[:,0]/Constants._au_to_meter), x[:,2]) * self.sigma_m(x[:,0]/Constants._au_to_meter) * self.trans_map(ra, dec, x[:,2], point) * self.instrument.field_of_view(ra, dec, x[:,2]) * x[:,0]
            list_exo_zodi[i] = integrate_jit_compiled_parts(exo_zodi, torch.tensor([[self.r_in*Constants._au_to_meter, self.r_out*Constants._au_to_meter], [-torch.pi, torch.pi], [point['wl_lo'], point['wl_hi']]]))
        self.exo_zodi.data = list_exo_zodi / self.distance**2


    def forward(self, data):
        # return the number of photons generated by the planet
        if len(self.exo_zodi) == 0:
            self.init_exo_zodi(data)

        return self.exo_zodi

class ExoZodiacalDustMatrix(ExoZodiacalDust):
    def __init__(self):
        super().__init__()
    def init_exo_zodi(self, data):
        vol_number = 500
        wl_number = 2
        radius_interp = torch.linspace(self.r_in*Constants._au_to_meter, self.r_out*Constants._au_to_meter, vol_number)
        psi_interp = torch.linspace(-torch.pi, torch.pi, vol_number)
        d_radius = (radius_interp[1] - radius_interp[0]) 
        d_psi = torch.abs(psi_interp[1] - psi_interp[0])

        r_mesh, psi_mesh = torch.meshgrid(radius_interp, psi_interp, indexing='ij')
        r_mesh = r_mesh.flatten()
        psi_mesh = psi_mesh.flatten()

        def exo_zodi(point):
            wl_interp = torch.linspace(point['wl_lo'], point['wl_hi'], wl_number)
            delta_wl = (point['wl_hi'] - point['wl_lo']) / wl_number
            def infin_zodi(radius, psi):
                theta = radius / self.distance
                ra, dec = self.trans_map.to_cartesian(theta, psi)
                interp_volume = d_psi / 2. * d_radius * (d_radius + 2*radius) * delta_wl
                return torch.sum(self.spectrum(self.temperature(radius/Constants._au_to_meter), wl_interp) * self.sigma_m(radius/Constants._au_to_meter) * self.trans_map(ra, dec, wl_interp, point) * self.instrument.field_of_view(ra, dec, wl_interp) * interp_volume)

            return torch.sum(torch.vmap(infin_zodi)(r_mesh, psi_mesh))

        list_exo_zodi = torch.zeros(len(data))
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            if i+chunk_size > len(data):
                list_exo_zodi[i:] = torch.vmap(exo_zodi)(data[i:])
            list_exo_zodi[i:i+chunk_size] = torch.vmap(exo_zodi)(data[i:i+chunk_size])

        self.exo_zodi.data = list_exo_zodi / self.distance**2