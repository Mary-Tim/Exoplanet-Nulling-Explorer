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

from nullingexplorer.utils import Constants as cons
from nullingexplorer.significance import PoissonSignificance

# Observation plan
obs_config = {
    'Spectrum':{
        #'Type': 'Equal',
        #'BinNumber': 1,
        #'Low': 0.9,
        #'High': 1.7,        # unit: micrometer
        'Type': 'Resolution',
        'R': 20,
        'Low': 3.,
        'High': 5.,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 1.,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 20.,  # unit: meter
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
        'nulling_depth': 1e-5,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
    }
}

sig_amp_config = {
    'Amplitude':{
        #'Proxima_b':{
        #    'Model': 'PlanetWithReflection',
        #    'Spectrum': 'BinnedBlackBody',
        #    'Parameters':
        #    {
        #        'au':        {'mean': 0.0485},
        #        'polar':          {'mean': 0.},
        #    },
        #    'Spectrum': {
        #        'Model': 'RelativeBlackBodySpectrum',
        #        'Parameters':
        #        {
        #            'r_radius':         {'mean': 1.1},
        #            'r_temperature':    {'mean': 0.821},
        #        },
        #    },
        #},
        'Proxima_d':{
            'Model': 'PlanetWithReflection',
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                'au':        {'mean': 0.02885},
                'polar':          {'mean': 0.},
            },
            'Spectrum': {
                'Model': 'RelativeBlackBodySpectrum',
                'Parameters':
                {
                    'r_radius':         {'mean': 0.81},
                    'r_temperature':    {'mean': 360./285.},
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
            'Model': 'StarBlackBodyMatrix',
        },
        'local_zodi':{
            'Model': 'LocalZodiacalDustMatrix',
        },
        'exo_zodi':{
            "Model": 'ExoZodiacalDustMatrix',
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

intg_time = np.linspace(0, 10000., 1000)

sig_poisson = PoissonSignificance()
sig_poisson.obs_config = obs_config

sig_poisson.sig_amp_config = sig_amp_config
sig_pe = sig_poisson.gen_sig_pe_single()

sig_poisson.bkg_amp_config = bkg_amp_config
bkg_pe = sig_poisson.gen_bkg_pe()

def sig_point(time):
    return sig_poisson.get_significance_single(sig_pe*time, bkg_pe*time)

significance = np.zeros(len(intg_time))

for i, time in tqdm(enumerate(intg_time), total=len(intg_time)):
    significance[i] = sig_point(time)

fig, ax = plt.subplots()
trans_map_cont = ax.plot(intg_time, significance, color='black')

ax.set_xlabel("Time / s")
ax.set_ylabel("Significance")

aimed_signifi = np.array([3,5,7], dtype=np.float64)

def aimed_time(signi):
    return np.power(signi/sig_point(1.0).cpu().detach().numpy(), 2)

for sigma in aimed_signifi:
    print(f"Observation time for {sigma:.0f}sigma in total: {aimed_time(sigma) * 360 / (3600):.2f} hour")

#plt.show()




