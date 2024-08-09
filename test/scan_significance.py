import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm

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
        'ObsNumber': 32,
        'IntegrationTime': 50*45,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 30.,  # unit: meter
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
        #'mars':{
        #    'Model': 'PlanetBlackBody',
        #    'Spectrum': 'InterpBlackBody',
        #    'Parameters':
        #    {
        #        'radius':         {'mean': 3389.5e3},
        #        'temperature':    {'mean': 210.},
        #        'ra':            {'mean': 80.},
        #        'dec':            {'mean': -129.24},
        #    },
        #},
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

scan_fov = [-200., 200.]
scan_num = 100

ra_line = np.linspace(scan_fov[0], scan_fov[1], scan_num)
dec_line = np.linspace(scan_fov[0], scan_fov[1], scan_num)

RA, DEC = np.meshgrid(ra_line, dec_line)

ra_array = RA.flatten()
dec_array = DEC.flatten()

sig_poisson = PoissonSignificance()
sig_poisson.obs_config = obs_config
sig_poisson.bkg_amp_config = bkg_amp_config

bkg_pe = sig_poisson.gen_bkg_pe()

def sig_point(ra, dec):
    sig_amp_config['Amplitude']['earth']['Parameters']['ra']['mean'] = ra
    sig_amp_config['Amplitude']['earth']['Parameters']['dec']['mean'] = dec
    sig_poisson.sig_amp_config = sig_amp_config
    sig_pe = sig_poisson.gen_sig_pe()
    return sig_poisson.get_significance(sig_pe, bkg_pe)

significance = np.zeros(len(ra_array))

for i, ra, dec in tqdm(zip(range(len(ra_array)), ra_array, dec_array), total=len(ra_array)):
    significance[i] = sig_point(ra, dec)

print(significance)
fig, ax = plt.subplots(figsize=(12,10))
levels = np.arange(0., 1.01*np.max(significance), 1.01*np.max(significance)/100.)
trans_map_cont = ax.contourf(RA, DEC, significance.reshape(scan_num, scan_num), levels=levels, cmap = plt.get_cmap("bwr"))

ax.set_xlabel("ra / mas")
ax.set_ylabel("dec / mas")
ax.axis('scaled')

cbar = fig.colorbar(trans_map_cont, format="%.2f")

print(f"Significance of earth: {sig_point(62.5, 78.1)}")

plt.show()




