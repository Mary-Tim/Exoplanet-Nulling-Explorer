
import torch
import torch.nn as nn

from .base_amplitude import BaseAmplitude
from .planet_spectrum import PlanetPolarCoordinates
from ..spectrum.planet_black_body import BlackBodySpectrum
from .star_spectrum import StarBlackBodyConstant
from nullingexplorer.utils import get_spectrum, get_transmission
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

class PlanetWithReflection(PlanetPolarCoordinates):
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
        ra, dec = self.trans_map.to_cartesian(self.au * self.au_to_mas / cons._radian_to_mas, self.polar)
        planet_emmision = super().forward(data)
        planet_reflection = self.star_bk(data) / (self.au * cons._au_to_meter)**2 \
                            * (self.spectrum.r_radius * self.spectrum.e_rad) ** 2 * self.reflection_scale \
                            * self.trans_map(ra, dec, data['wl_mid'], data) * self.instrument.field_of_view(ra, dec, data['wl_mid'])
        return planet_emmision + planet_reflection 