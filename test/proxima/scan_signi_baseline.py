import sys
sys.path.append('../..')

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

from nullingexplorer.utils import Constants as cons
from nullingexplorer.significance import PoissonSignificance

# Observation plan
obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 1.,
        'High': 2.,        # unit: micrometer
        #'Type': 'Equal',
        #'BinNumber': 1,
        #'Low': 0.9,
        #'High': 1.7,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 300.,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 10.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 0.6,   # Diameter of MiYin primary mirror [meter]
        'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
        'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
        'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
    }
}

sig_amp_config = {
    'Amplitude':{
        'earth':{
            #'Model': 'PlanetPolarCoordinates',
            'Model': 'PlanetWithReflection',
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                'au':        {'mean': 0.0485},
                'polar':          {'mean': 0.},
            },
            'Spectrum': {
                'Model': 'RelativeBlackBodySpectrum',
                'Parameters':
                {
                    'r_radius':         {'mean': 1.1},
                    'r_temperature':    {'mean': 0.821},
                },
            },
        },
    },
    'Instrument': 'MiYinTwoMirror',
    #'TransmissionMap': 'DualChoppedDestructive',
    'TransmissionMap': 'SingleBracewell',
    'Configuration':{
        'distance': 4.2 * cons._light_year_to_meter / cons._pc_to_meter,         # distance between Miyin and target [pc]
        'star_radius': 0.1542*cons._sun_radius,  # Star radius [kilometer]
        'star_temperature': 2992.,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': -62.67,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

bkg_amp_config = {
    'Amplitude':{
        'star':{
            'Model': 'StarBlackBodyConstant',
        },
        'local_zodi':{
            'Model': 'LocalZodiacalDustConstant',
        },
        'exo_zodi':{
            "Model": 'ExoZodiacalDustConstant',
        },
    },
    'Instrument': 'MiYinTwoMirror',
    #'TransmissionMap': 'DualChoppedDestructive',
    'TransmissionMap': 'SingleBracewell',
    'Configuration':{
        'distance': 4.2 * cons._light_year_to_meter / cons._pc_to_meter,         # distance between Miyin and target [pc]
        'star_radius': 0.1542*cons._sun_radius,  # Star radius [kilometer]
        'star_temperature': 2992.,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': -62.67,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

angular_separation = np.linspace(0.01, 0.1, 1000)

sig_poisson = PoissonSignificance()
#sig_poisson.obs_config = obs_config

sig_poisson.bkg_amp_config = bkg_amp_config
#bkg_pe = sig_poisson.gen_bkg_pe()

significance = np.zeros(len(angular_separation))
theta = 30. / 180. * np.pi
#baseline = np.array([3.], dtype=np.float32)
baseline = np.array([2., 3., 4., 5., 6.], dtype=np.float32)
#baseline = np.array([3., 5., 10., 20., 30.], dtype=np.float32)
#theta_array = np.array([0., 30., 45., 60., 90.], dtype=np.float32) / 180. * np.pi

fig, ax = plt.subplots()
for bl in tqdm(baseline):
    obs_config['Observation']['Baseline']['Value'] = bl
    sig_poisson.obs_config = obs_config
    bkg_pe = sig_poisson.gen_bkg_pe()
    for i, angular in enumerate(angular_separation):
        sig_amp_config['Amplitude']['earth']['Parameters']['au']['mean'] = angular
        sig_amp_config['Amplitude']['earth']['Parameters']['polar']['mean'] = theta
        sig_pe = sig_poisson.gen_sig_pe_single(sig_amp_config)
        significance[i] = sig_poisson.get_significance_single(sig_pe, bkg_pe)

    color = next(cycol)
    trans_map_cont = ax.plot(angular_separation, significance, color=color, label=f"{bl:.0f}m")

ax.set_xlabel("Distance [AU]")
ax.set_ylabel("Significance")

ax.axvline(x=0.0485, color='b', linestyle='--', label='Proxima b')
ax.axvline(x=0.02885, color='g', linestyle='--', label='Proxima d')

#ax.set_yscale('log')
ax.legend()

plt.show()




