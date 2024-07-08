import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd as atg
from scipy.optimize import minimize, basinhopping, brute
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.io import MiYinData, FitResult
from nullingexplorer.fitter import GaussianNLL
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.utils import get_amplitude

# Observation plan
phase_range=[0., np.pi*2]
phase_bins = 36000
spectrum_range = np.array([7., 18.], dtype=np.float64) * 1e-6
spectrum_bins = 30
integral_time = 278./100.

# Object config
earth_location=np.array([62.5, 78.1]) / cons._radian_to_mas
earth_temperature = 285.
earth_radius = 6371.e3
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

# Formation config
cfg.set_property('mirror_diameter', 3.5)
cfg.set_property('baseline', 15.)
cfg.set_property('ratio', 6.)

# Nulling depth config
depth_index = 5
cfg.set_property('nulling_depth', 0.)
#cfg.set_property('nulling_depth', 1./np.power(10, depth_index))

# Saving path
from datetime import datetime
import os
start_time = datetime.now()
output_path = f"results/Job_Signi_{start_time.strftime('%Y%m%d_%H%M%S')}"

phase = torch.tensor(np.repeat(np.linspace(phase_range[0], phase_range[1], phase_bins),spectrum_bins*2)).flatten()
wavelength = torch.tensor([np.repeat(np.linspace(spectrum_range[0], spectrum_range[1], spectrum_bins),2)]*phase_bins).flatten()
wl_width = torch.ones(phase_bins*spectrum_bins*2)*((spectrum_range[1]-spectrum_range[0])/spectrum_bins)
intg_time = torch.ones(phase_bins*spectrum_bins*2)*integral_time
mod = torch.tensor([np.array([1,-1])]*(phase_bins*spectrum_bins)).flatten()
photon_electron = torch.zeros(phase_bins*spectrum_bins*2)
pe_uncertainty = torch.zeros(phase_bins*spectrum_bins*2)

# 计算行星信号
signal = MiYinData(phase=phase, 
                 wavelength=wavelength, 
                 wl_width=wl_width, 
                 mod=mod, 
                 integral_time=intg_time, 
                 photon_electron=photon_electron, 
                 pe_uncertainty=pe_uncertainty, 
                 batch_size=[phase_bins*spectrum_bins*2])

class SignalAmplitude(BaseAmplitude):
    def __init__(self):
        super(SignalAmplitude, self).__init__()
        self.earth = PlanetBlackBody()

    def forward(self, data):
        return (self.earth(data)) * self.instrument(data)

signal_amp = SignalAmplitude()
signal.photon_electron = signal_amp(signal)

# 计算本底
background = signal.clone()

class BackgroundAmplitude(BaseAmplitude):
    def __init__(self):
        super(BackgroundAmplitude, self).__init__()
        self.star = StarBlackBodyFast()
        self.local_zodi = LocalZodiacalDustFast()
        self.exo_zodi = ExoZodiacalDustFast()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.star(data) + self.local_zodi(data) + \
                self.exo_zodi(data)) * self.instrument(data)

background_amp = BackgroundAmplitude()
background.photon_electron = background_amp(background)