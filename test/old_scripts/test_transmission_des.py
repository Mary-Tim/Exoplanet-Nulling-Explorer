import sys
sys.path.append('..')

import torch
import torch.nn as nn
from tensordict.prototype import tensorclass
from nullingexplorer.utils import get_transmission
from nullingexplorer import *
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda:0')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from scipy.interpolate import griddata

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor

fov = np.array([-200., 200.], dtype=np.float64) / cons._radian_to_mas
bins = 200

cfg.set_property('baseline', 5.)
cfg.set_property('ratio', 6)
cfg.set_property('mirror_diameter', 3.5)

#model = get_transmission('SingleBracewell')()
model = get_transmission('DualChoppedDestructive')()

ra = torch.tensor(np.linspace(fov[0], fov[1], bins))
dec = torch.tensor(np.linspace(fov[0], fov[1], bins))

ra_grid, dec_grid = torch.meshgrid(ra, dec, indexing='ij')
ra_grid_numpy = ra_grid.cpu().detach().numpy()
dec_grid_numpy = dec_grid.cpu().detach().numpy()

points = torch.stack((ra_grid.flatten(), dec_grid.flatten()), -1)
#data = {
#    'wavelength': torch.tensor(10.e-6),
#    'phi': torch.tensor(1.5*np.pi)
#}
phase = torch.tensor(0.)
wavelength = torch.tensor(10.e-6)
wl_width = torch.tensor(10e-7)
mod = torch.tensor(1)
data1 = MiYinData(phase=phase, wavelength=wavelength, wl_width=wl_width, mod=mod, batch_size=[])
trans_map1 = model(ra_grid, dec_grid, data1.wavelength, data1).cpu().detach().numpy()
data2 = MiYinData(phase=phase, wavelength=wavelength, wl_width=wl_width, mod=-mod, batch_size=[])
trans_map2 = model(ra_grid, dec_grid, data2.wavelength, data2).cpu().detach().numpy()
trans_map = trans_map1
#trans_map = trans_map1 - trans_map2
#print(len(trans_map))
#print(len(points))

draw_trans = griddata(points.cpu().detach().numpy(), trans_map.flatten(), (ra_grid_numpy, dec_grid_numpy), method='cubic')

fig, ax = plt.subplots()
levels = np.arange(-1, 1., 0.01)
#levels = np.arange(np.min(trans_map)*1.001, np.max(trans_map)*1.001, 0.01)
trans_map_cont = ax.contourf(ra_grid_numpy*cons._radian_to_mas, dec_grid_numpy*cons._radian_to_mas, trans_map, levels=levels, cmap = plt.get_cmap("bwr"))
ax.set_xlabel("ra / mas")
ax.set_ylabel("dec / mas")

cbar = fig.colorbar(trans_map_cont)

plt.show()