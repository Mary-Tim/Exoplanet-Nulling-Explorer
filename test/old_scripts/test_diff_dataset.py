import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from scipy.interpolate import griddata
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd as atg
from scipy.optimize import minimize, basinhopping, brute
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)
from tensordict.prototype import tensorclass

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

phase_range=[0., np.pi*2]
phase_bins = 360
spectrum_range = np.array([7., 18.], dtype=np.float64) * 1e-6
spectrum_bins = 30
integral_time = 278.
earth_location=np.array([62.5, 78.1]) / cons._radian_to_mas
#earth_location=np.array([100./np.sqrt(2), 100./np.sqrt(2)]) / cons._radian_to_mas
earth_temperature = 285.

cfg.set_property('mirror_diameter', 3.5)
cfg.set_property('baseline', 15.)
cfg.set_property('ratio', 6.)
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

# 设置ra, dec扫描范围
fov = np.array([-200., 200.], dtype=np.float64) / cons._radian_to_mas
fov_bins = 200

def significance(ndf=2, sig=-1000, bkg=-1000):
    delta_2ll = 2 * abs(sig-bkg)
    n_sigma = -stats.norm.ppf(stats.chi2.sf(delta_2ll,df=ndf,loc=0,scale=1)/2)
    return n_sigma

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor
    integral_time: torch.Tensor
    photon_electron: torch.Tensor
    pe_uncertainty: torch.Tensor

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
pe_uncertainty = torch.zeros(phase_bins*spectrum_bins*2)

data = MiYinData(phase=phase, 
                 wavelength=wavelength, 
                 wl_width=wl_width, 
                 mod=mod, 
                 integral_time=intg_time, 
                 photon_electron=photon_electron, 
                 pe_uncertainty=pe_uncertainty, 
                 batch_size=[phase_bins*spectrum_bins*2])

class Amplitude(BaseAmplitude):
    def __init__(self):
        super(Amplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.star = StarBlackBodyFast()
        self.local_zodi = LocalZodiacalDustFast()
        self.exo_zodi = ExoZodiacalDustFast()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        #return self.earth(data) * self.instrument(data)
        return (self.earth(data) + self.star(data) + self.local_zodi(data) + self.exo_zodi(data)) * self.instrument(data)


amp = Amplitude()
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(6371.e3)

data.photon_electron = torch.poisson(amp(data))
diff_data = data.reshape(phase_bins,spectrum_bins,2)[:,:,0]
diff_data.photon_electron = (data.select_mod(1).photon_electron - data.select_mod(-1).photon_electron).reshape(phase_bins,spectrum_bins)
diff_data.pe_uncertainty = torch.sqrt(data.select_mod(1).photon_electron + data.select_mod(-1).photon_electron).reshape(phase_bins,spectrum_bins)
diff_data.pe_uncertainty[diff_data.pe_uncertainty == 0] = 1e10

pe_np = diff_data.photon_electron.cpu().detach().numpy().reshape(phase_bins, spectrum_bins)
pe_err_np = diff_data.pe_uncertainty.cpu().detach().numpy().reshape(phase_bins, spectrum_bins)
phase_np = diff_data.phase.cpu().detach().numpy()
wavelength_np = diff_data.wavelength.cpu().detach().numpy()

# draw pe
fig, ax = plt.subplots(2,1, figsize=(24.,7.))

draw_data = ax[0].pcolormesh(phase_np, wavelength_np, pe_np, cmap = plt.get_cmap("bwr"))
ax[0].set_xlabel("phase / rad")
ax[0].set_ylabel("wavelength / m")
cbar = fig.colorbar(draw_data, label="Photon electron", aspect=5, pad=0.01)

draw_err = ax[1].pcolormesh(phase_np, wavelength_np, pe_err_np, cmap = plt.get_cmap("bwr"))
ax[1].set_xlabel("phase / rad")
ax[1].set_ylabel("wavelength / m")
cbar = fig.colorbar(draw_err, label="Uncertainty", aspect=5, pad=0.01)

import tempfile
#file_h5 = tempfile.NamedTemporaryFile()
with open('diff_data.hdf5', "w") as file_h5:
    diff_data.detach().to_h5(file_h5.name, compression="gzip", compression_opts=9)
    file_h5.close()

#plt.show()
