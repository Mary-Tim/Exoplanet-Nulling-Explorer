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

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.io import MiYinData

phase_range=[0., np.pi*2]
phase_bins = 360
spectrum_range = np.array([7., 18.], dtype=np.float64) * 1e-6
spectrum_bins = 30
integral_time = 278.*2
earth_location=np.array([100./np.sqrt(2), 100./np.sqrt(2)]) / cons._radian_to_mas
earth_temperature = 285.

cfg.set_property('baseline', 15.)
cfg.set_property('ratio', 6.)
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

phase = torch.tensor(np.repeat(np.linspace(phase_range[0], phase_range[1], phase_bins),spectrum_bins)).flatten()
wavelength = torch.tensor([np.linspace(spectrum_range[0], spectrum_range[1], spectrum_bins)]*phase_bins).flatten()
wl_width = torch.ones(phase_bins*spectrum_bins)*((spectrum_range[1]-spectrum_range[0])/spectrum_bins)
intg_time = torch.ones(phase_bins*spectrum_bins)*integral_time
mod = torch.tensor([np.array([1])]*(phase_bins*spectrum_bins)).flatten()
photon_electron = torch.zeros(phase_bins*spectrum_bins)
pe_uncertainty = torch.zeros(phase_bins*spectrum_bins)

data = MiYinData(phase=phase, wavelength=wavelength, wl_width=wl_width, mod=mod, integral_time=intg_time, photon_electron=photon_electron, pe_uncertainty=pe_uncertainty, batch_size=[phase_bins*spectrum_bins])

class Amplitude(BaseAmplitude):
    def __init__(self):
        super(Amplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.star = StarBlackBodyFast()
        self.local_zodi = LocalZodiacalDustFast()
        self.exo_zodi = ExoZodiacalDustFast()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        #return (self.earth(data)) * self.instrument(data)
        return (self.earth(data)+ self.star(data) + self.local_zodi(data) + self.exo_zodi(data)) * self.instrument(data)

amp = Amplitude()
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(6371.e3)

#data.photon_electron = amp(data)
data.photon_electron = torch.poisson(amp(data))
#print(data.photon_electron)
data.draw(path='fig',draw_err=False, show=True, save=True)

#import tempfile
##file_h5 = tempfile.NamedTemporaryFile()
#with open('diff_o3_data.h5', "w") as file_h5:
#    data.detach().to_h5(file_h5.name, compression="gzip", compression_opts=9)
#    file_h5.close()