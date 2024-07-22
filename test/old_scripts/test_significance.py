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
from nullingexplorer.generator import AmplitudeCreator
from nullingexplorer.io import DataHandler
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

# Observation plan
obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 4.,
        'High': 18.5,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 100,  # unit: second
        'ObsMode': [1, -1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 15.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 4,   # Diameter of MiYin primary mirror [meter]
        'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
        'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
        'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
    }
}

sig_amp_config = {
    'Amplitude':{
        'earth':{
            'Model': 'PlanetBlackBody',
            'Spectrum': 'InterpBlackBody',
            'Parameters':
            {
                'radius':         {'mean': 6371.e3},
                'temperature':    {'mean': 285.},
                'ra':            {'mean': 62.5},
                'dec':            {'mean': 78.1},
            },
        },
    },
    'Instrument': 'MiYinBasicType',
    'TransmissionMap': 'DualChoppedDestructive',
    'Configuration':{
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

bkg_amp_config = {
    'Amplitude':{
        'star':{
            'Model': 'StarBlackBodyMatrix',
        },
        'local_zodi':{
            'Model': 'LocalZodiacalDustMatrix',
        },
        'exo_zodi':{
            "Model": 'ExoZodiacalDustMatrix',
        },
    },
    'Instrument': 'MiYinBasicType',
    'TransmissionMap': 'DualChoppedDestructive',
    'Configuration':{
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

# 创建观测计划

# 计算行星Chopped信号
obs_creator = ObservationCreator()
obs_creator.load(obs_config)
sig_data = obs_creator.generate()

sig_amp = AmplitudeCreator(config=sig_amp_config)
sig_data['photon_electron'] = torch.poisson(sig_amp(sig_data))
data_handler = DataHandler(sig_data)
sig_data = data_handler.diff_data(obs_creator)

# 计算本底
obs_config['Observation']['ObsMode'] = [1]
obs_creator.load(obs_config)
bkg_data = obs_creator.generate()

bkg_amp = AmplitudeCreator(config=bkg_amp_config)
bkg_data['photon_electron'] = torch.poisson(bkg_amp(bkg_data))

# Reshape
sig_pe = sig_data['photon_electron'].reshape(obs_creator.obs_num, obs_creator.spec_num).t()
bkg_pe = bkg_data['photon_electron'].reshape(obs_creator.obs_num, obs_creator.spec_num).t()

def cal_SNR_wl(sig, bg):
    return torch.sum(torch.sqrt(sig**2)) / torch.sqrt(2 * torch.sum(bg) + torch.sum(torch.sqrt(sig**2)))
    #return torch.sum(torch.sqrt(sig**2)) / torch.sqrt(2 * (torch.sum(bg) + torch.sum(torch.sqrt(sig_o3**2))))

SNR_wl = torch.vmap(cal_SNR_wl)(sig_pe, bkg_pe)

SNR = torch.sqrt(torch.sum(SNR_wl**2))
print(f'SNR_wl: {SNR_wl}')
print(f"SNR: {SNR:.3f}")