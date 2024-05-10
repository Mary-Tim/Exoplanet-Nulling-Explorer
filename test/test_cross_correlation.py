import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from scipy.interpolate import griddata

import torch
import torch.nn as nn
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)
from tensordict.prototype import tensorclass

from nullingexplorer.amplitude import *
from nullingexplorer.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

phase_range=[0., np.pi*2]
phase_bins = 360
spectrum_range = np.array([5., 25.], dtype=np.float64) * 1e-6
spectrum_bins = 30
integral_time = 540.
earth_location=np.array([100./np.sqrt(2), 100./np.sqrt(2)]) / cons._radian_to_mac
earth_temperature = 285.
mars_location=np.array([150., 0.]) / cons._radian_to_mac
mars_temperature = 210.
venus_location=np.array([-50./np.sqrt(2), 50./np.sqrt(2)]) / cons._radian_to_mac
venus_temperature = 737.

cfg.set_property('baseline', 15.)
cfg.set_property('ratio', 6.)
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor
    integral_time: torch.Tensor
    photon_electron: torch.Tensor

    def select_mod(self, mod):
        return self[self.mod==mod]

    def select_data(self, key, val):
        return self[getattr(self, key)==val]

    def nparray(self, key):
        return getattr(self, key).cpu().detach().numpy()

    def get_bins(self, key):
        return torch.unique(getattr(self, key))

    def get_bin_number(self, key):
        return torch.tensor(len(torch.unique(getattr(self, key))))

phase = torch.tensor(np.repeat(np.linspace(phase_range[0], phase_range[1], phase_bins),spectrum_bins*2)).flatten()
wavelength = torch.tensor([np.repeat(np.linspace(spectrum_range[0], spectrum_range[1], spectrum_bins),2)]*phase_bins).flatten()
wl_width = torch.ones(phase_bins*spectrum_bins*2)*((spectrum_range[1]-spectrum_range[0])/spectrum_bins)
intg_time = torch.ones(phase_bins*spectrum_bins*2)*integral_time
mod = torch.tensor([np.array([1,-1])]*(phase_bins*spectrum_bins)).flatten()
photon_electron = torch.zeros(phase_bins*spectrum_bins*2)

data = MiYinData(phase=phase, wavelength=wavelength, wl_width=wl_width, mod=mod, integral_time=intg_time, photon_electron=photon_electron, batch_size=[phase_bins*spectrum_bins*2])

class Amplitude(BaseAmplitude):
    def __init__(self):
        super(Amplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.mars = PlanetBlackBody()
        self.venus = PlanetBlackBody()
        self.star = StarBlackBodyFast()
        self.local_zodi = LocalZodiacalDustFast()
        self.exo_zodi = ExoZodiacalDustFast()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        #return (self.earth(data)+ self.mars(data) + self.venus(data)) * self.instrument(data)
        return (self.earth(data)+ self.star(data) + self.local_zodi(data) + self.exo_zodi(data)) * self.instrument(data)

amp = Amplitude()
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(6371.e3)
amp.mars.ra.data = torch.tensor(mars_location[0])
amp.mars.dec.data = torch.tensor(mars_location[1])
amp.mars.temperature.data = torch.tensor(mars_temperature)
amp.mars.radius.data = torch.tensor(3389.5e3)
amp.venus.ra.data = torch.tensor(venus_location[0])
amp.venus.dec.data = torch.tensor(venus_location[1])
amp.venus.temperature.data = torch.tensor(venus_temperature)
amp.venus.radius.data = torch.tensor(6051.8e3)

#data.photon_electron = amp(data)
data.photon_electron = torch.poisson(amp(data))
print(data.photon_electron)

diff_data = data.reshape(phase_bins,spectrum_bins,2)[:,:,0]
diff_data.photon_electron = (data.select_mod(1).photon_electron - data.select_mod(-1).photon_electron).reshape(phase_bins,spectrum_bins)
print(diff_data)

# Draw phase-wavelength 2D diagram
points = torch.stack((diff_data.phase.flatten(), diff_data.wavelength.flatten()), -1).cpu().detach().numpy()
phase_np = diff_data.nparray('phase')
wavelength_np = diff_data.nparray('wavelength')
pe_np = diff_data.nparray('photon_electron').flatten()

draw_pe = griddata(points, pe_np, (phase_np, wavelength_np), method='cubic')

#fig, ax = plt.subplots()
#print(f"length of pe_np:{len(pe_np)}")
##levels = np.arange(np.min(pe_np), np.max(pe_np), 0.001)
#trans_map_cont = ax.contourf(phase_np, wavelength_np, draw_pe, cmap = plt.get_cmap("bwr"))
#ax.set_xlabel("phase / rad")
#ax.set_ylabel("wavelength / m")

#cbar = fig.colorbar(trans_map_cont)

#plt.show()

# Cross correlation plot
from nullingexplorer.transmission import DualChoppedDifferential

fov = np.array([-200., 200.], dtype=np.float64) / cons._radian_to_mac
fov_bins = 200

ra = torch.tensor(np.linspace(fov[0], fov[1], fov_bins))
dec = torch.tensor(np.linspace(fov[0], fov[1], fov_bins))

ra_grid, dec_grid = torch.meshgrid(ra, dec, indexing='ij')
ra_grid_numpy = ra_grid.cpu().detach().numpy()
dec_grid_numpy = dec_grid.cpu().detach().numpy()

points = torch.stack((ra_grid.flatten(), dec_grid.flatten()), -1)

class CrossCorrelation(nn.Module):
    def __init__(self) -> None:
        super(CrossCorrelation, self).__init__()
        self.trans_map = DualChoppedDifferential()

    def forward(self, ra, dec, data):
        return data.photon_electron*self.trans_map(ra, dec, data.wavelength, data)

cs_model = CrossCorrelation()
cs_plot = torch.stack([torch.sum(cs_model(pt[0], pt[1], diff_data.flatten())) for pt in points]).cpu().detach().numpy()
#cs_plot = np.log(cs_plot)

draw_cs = griddata(points.cpu().detach().numpy(), cs_plot, (ra_grid_numpy, dec_grid_numpy), method='cubic')

fig, ax = plt.subplots()
levels = np.arange(np.min(cs_plot)*1.005, np.max(cs_plot)*1.005, (np.max(cs_plot)-np.min(cs_plot))/100.)
cs_cont = ax.contourf(ra_grid_numpy*cons._radian_to_mac, dec_grid_numpy*cons._radian_to_mac, draw_cs, levels=levels, cmap = plt.get_cmap("bwr_r"))
ax.set_xlabel("ra / mas")
ax.set_ylabel("dec / mas")

cbar = fig.colorbar(cs_cont)

plt.savefig('fig/cross_correlation.pdf')
plt.show()