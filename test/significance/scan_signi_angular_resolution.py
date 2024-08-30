import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm
from scipy.interpolate import interpn

from itertools import cycle
cycol = cycle('bgrcmk')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.significance import PoissonSignificance

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
            #'Type': 'Linear',
            #'Low': 10.,  # unit: meter
            #'High': 50.,  # unit: meter
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
    'Instrument': {'Model': 'MiYinBasicType'},
    'TransmissionMap': {'Model': 'DualChoppedDestructive'},
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
    'Instrument': {'Model': 'MiYinBasicType'},
    'TransmissionMap': {'Model': 'DualChoppedDestructive'},
    'Configuration':{
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

angular_separation = np.linspace(1., 1000., 200)

sig_poisson = PoissonSignificance()
#sig_poisson.obs_config = obs_config

sig_poisson.bkg_amp_config = bkg_amp_config
#bkg_pe = sig_poisson.gen_bkg_pe()

significance = np.zeros(len(angular_separation))
theta = 30. / 180. * np.pi
baseline = np.array([10., 15., 30., 50], dtype=np.float32)
#theta_array = np.array([0., 30., 45., 60., 90.], dtype=np.float32) / 180. * np.pi

fig, ax = plt.subplots()
for bl in tqdm(baseline):
    obs_config['Observation']['Baseline']['Value'] = bl
    sig_poisson.obs_config = obs_config
    bkg_pe = sig_poisson.gen_bkg_pe()
    for i, angular in enumerate(angular_separation):
        sig_amp_config['Amplitude']['earth']['Parameters']['ra']['mean'] = angular * np.cos(theta)
        sig_amp_config['Amplitude']['earth']['Parameters']['dec']['mean'] = angular * np.sin(theta)
        sig_pe = sig_poisson.gen_sig_pe(sig_amp_config)
        significance[i] = sig_poisson.get_significance(sig_pe, bkg_pe)

    color = next(cycol)
    trans_map_cont = ax.plot(angular_separation, significance, color=color, label=f"{bl:.0f}m")

ax.set_xlabel("Angular / mas")
ax.set_ylabel("Significance")

ax.legend()

plt.show()




