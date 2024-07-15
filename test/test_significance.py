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
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.generator import ObservationCreator
from nullingexplorer.io import DataHandler
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

# Observation plan
obs_config = {
        'Spectrum':
            {
            'Type': 'Resolution',
            'R': 20,
            'Low': 4.,
            'High': 18.5,        # unit: micrometer
            },
        'Observation':
            {
            'ObsNumber': 360,
            'IntegrationTime': 100,  # unit: second
            'ObsMode': [1, -1],  # [1] or [-1] or [1, -1]
            'Phase':
                {
                'Start' : 0.,
                'Stop': 360.,   # unit: degree
                },
            'Baseline':
                {
                'Type': 'Constant',
                'Value': 15.,  # unit: meter
                },
            },
        }

# Object config
earth_location=np.array([62.5, 78.1]) / cons._radian_to_mas
earth_temperature = 285.
earth_radius = 6371.e3
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

# Formation config
cfg.set_property('mirror_diameter', 3.5)
cfg.set_property('ratio', 6.)

# Saving path
from datetime import datetime
import os
start_time = datetime.now()
output_path = f"results/Job_Signi_{start_time.strftime('%Y%m%d_%H%M%S')}"

# 计算行星Chopped信号
obs_creator = ObservationCreator()
obs_creator.load(obs_config)
data = obs_creator.generate()

cfg.set_property('trans_map', 'DualChoppedDestructive')
class SignalAmplitude(BaseAmplitude):
    def __init__(self):
        super(SignalAmplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.earth(data)) * self.instrument(data)

signal_amp = SignalAmplitude()
signal_amp.earth.ra.data = torch.tensor(earth_location[0])
signal_amp.earth.dec.data = torch.tensor(earth_location[1])
signal_amp.earth.temperature.data = torch.tensor(earth_temperature)
signal_amp.earth.radius.data = torch.tensor(earth_radius)
data['photon_electron'] = signal_amp(data)
data_handler = DataHandler(data)
signal_data = data_handler.diff_data(obs_creator)

# 计算本底
obs_config['Observation']['ObsMode'] = [1]
obs_creator.load(obs_config)
bg_data = obs_creator.generate()

class BackgroundAmplitude(BaseAmplitude):
    def __init__(self):
        super(BackgroundAmplitude, self).__init__()
        self.star = StarBlackBodyMatrix()
        self.local_zodi = LocalZodiacalDustMatrix()
        self.exo_zodi = ExoZodiacalDustMatrix()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.star(data) + self.local_zodi(data) + \
                self.exo_zodi(data)) * self.instrument(data)

bg_amp = BackgroundAmplitude()
bg_data['photon_electron'] = bg_amp(bg_data)

# Reshape
signal_pe = signal_data['photon_electron'].reshape(obs_creator.obs_num, obs_creator.spec_num).t()
bg_pe = bg_data['photon_electron'].reshape(obs_creator.obs_num, obs_creator.spec_num).t()

def cal_SNR_wl(sig, bg):
    return torch.sum(torch.sqrt(sig**2)) / torch.sqrt(2 * torch.sum(bg) + torch.sum(torch.sqrt(sig**2)))
    #return torch.sum(torch.sqrt(sig**2)) / torch.sqrt(2 * (torch.sum(bg) + torch.sum(torch.sqrt(sig_o3**2))))

SNR_wl = torch.vmap(cal_SNR_wl)(signal_pe, bg_pe)

SNR = torch.sqrt(torch.sum(SNR_wl**2))
print(f'SNR_wl: {SNR_wl}')
print(f"SNR: {SNR:.3f}")