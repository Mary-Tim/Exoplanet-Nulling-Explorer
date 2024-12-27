import torch
import torch.nn as nn

from .base_orbit import BaseOrbit
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.utils import get_transmission

class StaticCartesianCoordinates(BaseOrbit):
    def __init__(self):
        super().__init__()
        # Free parameters
        self.ra  = nn.Parameter(torch.tensor(50.)) # Right ascension relative to the star (unit: mas)
        self.dec = nn.Parameter(torch.tensor(50.)) # Declination relative to the star (unit: mas)

        # Boundary of parameters
        self.boundary = {
            'ra': torch.tensor([-1000., 1000.]),
            'dec': torch.tensor([-1000., 1000.]),
        } 

    def forward(self, data):
        return self.ra / cons._radian_to_mas, self.dec / cons._radian_to_mas

class StaticPolarCoordinates(BaseOrbit):
    def __init__(self):
        super().__init__()
        # Transmission map
        self.trans_map = get_transmission(cfg.get_property('trans_map'))()
        # Constant parameters
        self.register_buffer('distance', cfg.get_property('distance') * cons._pc_to_meter) # Distance between target and format (unit: pc)
        self.register_buffer('au_to_mas', (cons._au_to_meter) / (self.distance) * cons._radian_to_mas)

        # Free parameters
        self.au  = nn.Parameter(torch.tensor(1.)) # Relative angular separation relative to the star (unit: dimensionless)
        self.polar    = nn.Parameter(torch.tensor(1.)) # Relative declination relative to the star (unit: dimensionless)

        # Boundary of parameters
        self.boundary = {
            'au': torch.tensor([0.1, 5.]),
            'polar': torch.tensor([0., 2*torch.pi]),
        }

    def forward(self, data):
        ra, dec = self.trans_map.to_cartesian(self.au * self.au_to_mas / cons._radian_to_mas, self.polar)
        return ra, dec