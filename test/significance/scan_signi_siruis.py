import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm
from scipy.interpolate import interpn

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.significance import PoissonSignificance

# Observation plan
obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 0.9,
        'High': 1.8,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 1,  # unit: second
        'ObsMode': [1, -1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 10.,  # unit: meter
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
        'mirror_diameter': 0.2,   # Diameter of MiYin primary mirror [meter]
        'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
        'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
        'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
        'fov_scale': 100000.
    }
}

sig_amp_config = {
    'Amplitude':{
        'earth':{
            'Model': 'PlanetBlackBody',
            'Spectrum': 'InterpBlackBody',
            'Parameters':
            {
                'radius':         {'mean': 6371.e3*1.4},
                'temperature':    {'mean': 25000},
                'ra':             {'mean': 7800.},
                'dec':            {'mean': 0.},
            },
        },
    },
    'Instrument': {'Model': 'MiYinBasicType'},
    'TransmissionMap': {'Model': 'DualChoppedDestructive'},
    'Configuration':{
        'distance': 2.637,         # distance between Miyin and target [pc]
        'star_radius': 695500*1.7,  # Star radius [kilometer]
        'star_temperature': 9940,   # Star temperature [Kelvin]
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
    #'Instrument': {'Model': 'MiYinBasicType'},
    'Instrument': {'Model': 'MiYinBasicType'},
    'TransmissionMap': {'Model': 'DualChoppedDestructive'},
    'Configuration':{
        'distance': 2.637,         # distance between Miyin and target [pc]
        'star_radius': 695500*1.7,  # Star radius [kilometer]
        'star_temperature': 9940,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

intg_time = np.linspace(1., 200., 100)

sig_poisson = PoissonSignificance()
sig_poisson.obs_config = obs_config

sig_poisson.sig_amp_config = sig_amp_config
sig_pe = sig_poisson.gen_sig_pe()

sig_poisson.bkg_amp_config = bkg_amp_config
bkg_pe = sig_poisson.gen_bkg_pe()
print(sig_pe)
print(bkg_pe)

def sig_point(time):
    return sig_poisson.get_significance(sig_pe*time, bkg_pe*time)

significance = np.zeros(len(intg_time))

for i, time in tqdm(enumerate(intg_time), total=len(intg_time)):
    significance[i] = sig_point(time)

fig, ax = plt.subplots()
trans_map_cont = ax.plot(intg_time, significance, color='black')

ax.set_xlabel("Time / s")
ax.set_ylabel("Significance")

aimed_signifi = 10.

def aimed_time(signi):
    return np.power(signi/sig_point(1.0).cpu().detach().numpy(), 2)

print(f"Integral time for 10$\,\sigma$: {aimed_time(aimed_signifi):.2f} s")

plt.show()




