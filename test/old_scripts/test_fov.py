import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.instrument import MiYinBasicType

cfg.set_property('mirror_diameter', 2.)
#sigmod = nn.Sigmoid()
wavelength = torch.tensor(5e-6)

instrument = MiYinBasicType()

#FoV = lambda mas: 1-sigmod(mas+wavelength/mirror_radius)
#FoV = lambda mas: 0.5-sigmod(mas-wavelength/mirror_radius*cons._radian_to_mas)/2.

#FoV = lambda mas: 0.5 * (1 - sigmod(mas*cons._radian_to_mas-wavelength/(mirror_radius*2)*cons._radian_to_mas))
#FoV = lambda mas: 0.5*(1 - sigmod(mas-wavelength/(mirror_radius*2)*cons._radian_to_mas))
#print(f"{wavelength/mirror_radius*cons._radian_to_mas:.05f}")
#print(f"{wavelength/(mirror_radius*2)}")

ra = torch.tensor(np.linspace(0., 3e-6, 500))
dec = torch.tensor(0.)
plt.plot(ra.cpu().detach().numpy()*cons._radian_to_mas, instrument.field_of_view(ra, dec, wavelength).cpu().detach().numpy(), 'r-', label='fit')

plt.show()