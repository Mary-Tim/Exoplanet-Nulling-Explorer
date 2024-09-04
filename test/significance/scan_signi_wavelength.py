import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
import h5py
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
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                'radius':         {'mean': 6371.e3},
                'temperature':    {'mean': 285.},
                'ra':            {'mean': 100.},
                'dec':            {'mean': 0.},
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
            'Buffers': {
                'vol_number': 50,
                'wl_number': 2   
            }
        },
        'local_zodi':{
            'Model': 'LocalZodiacalDustMatrix',
            'Buffers': {
                'vol_number': 100,
            }
        },
        'exo_zodi':{
            "Model": 'ExoZodiacalDustMatrix',
            'Buffers': {
                'vol_number': 50,
                'wl_number': 2   
            }
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

wl_lo_range = np.array([1., 8.])
wl_hi_range = np.array([12., 30.])
scan_num = 10

wl_lo_line = np.linspace(wl_lo_range[0], wl_lo_range[1], scan_num, endpoint=True)
wl_hi_line = np.linspace(wl_hi_range[0], wl_hi_range[1], scan_num, endpoint=True)

WLLO, WLHI = np.meshgrid(wl_lo_line, wl_hi_line)

wl_lo_array = WLLO.flatten()
wl_hi_array = WLHI.flatten()

sig_poisson = PoissonSignificance()
sig_poisson.sig_amp_config = sig_amp_config
sig_poisson.bkg_amp_config = bkg_amp_config

def sig_point(wl_lo, wl_hi):
    obs_config['Spectrum']['Low'] = wl_lo
    obs_config['Spectrum']['Low'] = wl_hi
    print(f"Wl_lo: {wl_lo:.3f},\tWl_hi: {wl_hi:.3f}")
    sig_poisson.obs_config = obs_config
    sig_pe = sig_poisson.gen_sig_pe()
    bkg_pe = sig_poisson.gen_bkg_pe()
    return sig_poisson.get_significance(sig_pe, bkg_pe)

significance = np.zeros(len(wl_lo_array))

for i, wl_lo, wl_hi in tqdm(zip(range(len(wl_lo_array)), wl_lo_array, wl_hi_array), total=len(wl_lo_array)):
    significance[i] = sig_point(wl_lo, wl_hi)

with h5py.File(f"result_scan_wavelength.hdf5", 'w') as file:
    file.create_dataset('scan_num', data=scan_num)
    file.create_dataset('wl_lo', data=wl_lo_array)
    file.create_dataset('wl_hi', data=wl_hi_array)
    file.create_dataset('significance', data=significance)

fig, ax = plt.subplots()
levels = np.geomspace(0.001, 1.01*np.max(significance), 1000)
trans_map_cont = ax.contourf(WLLO, WLHI, significance.reshape(scan_num, scan_num), levels=levels, cmap = plt.get_cmap("bwr"), norm='log')

ax.set_xlabel("Low / $\\rm{{\\mu m}}$")
ax.set_ylabel("High / $\\rm{{\\mu m}}$")

cbar = fig.colorbar(trans_map_cont, format="%.2f")


plt.show()




