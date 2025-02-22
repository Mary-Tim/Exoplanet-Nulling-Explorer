import torch
import torch.nn as nn

from .base_amplitude import BaseAmplitude, BasePlanet
from nullingexplorer.utils import get_spectrum, get_transmission, get_orbit
from nullingexplorer.model.spectrum.planet_black_body import BlackBodySpectrum
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

class PlanetCartesianCoordinates(BasePlanet):
    def __init__(self, spec='BlackBodySpectrum'):
        super(PlanetCartesianCoordinates, self).__init__()
        # Surface Spectrum of the Planet
        self.spectrum = get_spectrum(spec)()
        #self.trans_map = SingleBracewell()
        self.trans_map = get_transmission(cfg.get_property('trans_map'))(is_planet=True)
        self.instrument = MiYinBasicType()
        # Free parameters
        self.ra  = nn.Parameter(torch.tensor(50.)) # Right ascension relative to the star (unit: mas)
        self.dec = nn.Parameter(torch.tensor(50.)) # Declination relative to the star (unit: mas)

        # Boundary of parameters
        self.boundary = {
            'ra': torch.tensor([-1000., 1000.]),
            'dec': torch.tensor([-1000., 1000.]),
        } 

    def forward(self, data):
        # return the number of photons generated by the planet
        return self.spectrum(data) * self.trans_map(self.ra / cons._radian_to_mas, self.dec / cons._radian_to_mas, data['wl_mid'], data) * \
               self.instrument.field_of_view(self.ra / cons._radian_to_mas, self.dec / cons._radian_to_mas, data['wl_mid'])

class PlanetPolarCoordinates(BasePlanet):
    def __init__(self, spec='BinnedBlackBody'):
        super(PlanetPolarCoordinates, self).__init__()
        # Surface Spectrum of the Planet
        self.spectrum = get_spectrum(spec)()
        self.trans_map = get_transmission(cfg.get_property('trans_map'))(is_planet=True)
        self.instrument = MiYinBasicType()
        # Constant parameters
        self.register_buffer('distance', cfg.get_property('distance') * cons._pc_to_meter) # Distance between target and format (unit: pc)
        #self.register_buffer('e_mas', torch.tensor(100.) / cons._radian_to_mas) # Earth-Sun angular separation at 10 pc (unit: mas)
        self.register_buffer('au_to_mas', (cons._au_to_meter) / (self.distance) * cons._radian_to_mas)

        # Free parameters
        self.au  = nn.Parameter(torch.tensor(1.)) # Relative angular separation relative to the star (unit: dimensionless)
        self.polar    = nn.Parameter(torch.tensor(1.)) # Relative declination relative to the star (unit: dimensionless)
        #self.r_angular  = nn.Parameter(torch.tensor(1.)) # Relative angular separation relative to the star (unit: dimensionless)
        #self.r_polar    = nn.Parameter(torch.tensor(1.)) # Relative declination relative to the star (unit: dimensionless)

        # Boundary of parameters
        self.boundary = {
            'au': torch.tensor([0.1, 5.]),
            'polar': torch.tensor([0., 2*torch.pi]),
        }

    def forward(self, data):
        # return the number of photons generated by the planet
        #return self.spectrum(data) * self.trans_map(self.r_angular * self.e_mas, self.r_polar, data['wl_mid'], data) * \
        #       self.instrument.field_of_view_polar(self.r_angular * self.e_mas, data['wl_mid'])
        ra, dec = self.trans_map.to_cartesian(self.au * self.au_to_mas / cons._radian_to_mas, self.polar)
        return self.spectrum(data) * self.trans_map(ra, dec, data['wl_mid'], data) * \
               self.instrument.field_of_view(ra, dec, data['wl_mid'])

class PlanetSpectrum(BasePlanet):
    def __init__(self, spec='BinnedBlackBody', orbit='StaticPolarCoordinates'):
        super().__init__()

        self.spectrum = get_spectrum(spec)()
        self.orbit = get_orbit(orbit)()
        self.trans_map = get_transmission(cfg.get_property('trans_map'))(is_planet=True)
        self.instrument = MiYinBasicType()

    def forward(self, data):
        ra, dec = self.orbit(data)
        return self.spectrum(data) * self.trans_map(ra, dec, data['wl_mid'], data) * \
               self.instrument.field_of_view(ra, dec, data['wl_mid'])

class PlanetWithReflection(PlanetSpectrum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('reflection_scale', torch.tensor(0.2))
        self.register_buffer('star_radius', cfg.get_property('star_radius') * 1e3) # Star radius (unit: kilometer)
        self.register_buffer('star_temperature', cfg.get_property('star_temperature')) # Star radius (unit: kilometer)

        self.star_bk = BlackBodySpectrum()
        del self.star_bk.radius
        del self.star_bk.temperature
        self.star_bk.register_buffer('radius', self.star_radius.data)
        self.star_bk.register_buffer('temperature', self.star_temperature.data)

    def forward(self, data):
        planet_emmision = super().forward(data)
        ra, dec = self.orbit(data)
        planet_reflection = self.star_bk(data) / (self.orbit.au * cons._au_to_meter)**2 \
                            * (self.spectrum.r_radius * self.spectrum.e_rad) ** 2 * self.reflection_scale \
                            * self.trans_map(ra, dec, data['wl_mid'], data) * self.instrument.field_of_view(ra, dec, data['wl_mid'])
        return planet_emmision + planet_reflection 
