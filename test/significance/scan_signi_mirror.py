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

from nullingexplorer.significance import PoissonSignificance

# Observation plan
obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 5.,
        'High': 17,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 1.,  # unit: second
        'ObsMode': [1, -1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 30.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 3.5,   # Diameter of MiYin primary mirror [meter]
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
                'ra':            {'mean': 100.},
                'dec':            {'mean': 0.},
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
            'Model': 'StarBlackBodyConstant',
        },
        'local_zodi':{
            'Model': 'LocalZodiacalDustConstant',
        },
        'exo_zodi':{
            "Model": 'ExoZodiacalDustConstant',
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

obs_number = 360.
max_time = 200.*3600/obs_number
time_array = np.linspace(0., max_time, 1000)

sig_poisson = PoissonSignificance()

#sig_poisson.bkg_amp_config = bkg_amp_config
#bkg_pe = sig_poisson.gen_bkg_pe()

theta = 0. / 180. * np.pi
mirror_diameter = np.array([1., 2., 3., 4.], dtype=np.float32)
#distance = np.array([2., 3., 20., 30.], dtype=np.float32)
#theta_array = np.array([0., 30., 45., 60., 90.], dtype=np.float32) / 180. * np.pi

obs_config['Observation']['Baseline']['Value'] = 30.
sig_amp_config['Amplitude']['earth']['Parameters']['ra']['mean'] = 100. * np.cos(theta)
sig_amp_config['Amplitude']['earth']['Parameters']['dec']['mean'] = 100. * np.sin(theta)

time_at_ten_sigma = np.zeros(len(mirror_diameter))
fig, ax = plt.subplots()
for imd, md in tqdm(enumerate(mirror_diameter), total=len(mirror_diameter)):
    obs_config['Configuration']['mirror_diameter'] = md
    sig_poisson.obs_config = obs_config
    bkg_pe = sig_poisson.gen_bkg_pe(bkg_amp_config)
    sig_pe = sig_poisson.gen_sig_pe(sig_amp_config)
    significance = np.zeros(len(time_array))
    for i, time in enumerate(time_array):
        significance[i] = sig_poisson.get_significance(sig_pe*time, bkg_pe*time)

    color = next(cycol)
    #trans_map_cont = ax.plot(time_array * 10. / dt, significance, color=color, label=f"{dt:.0f}pc")
    trans_map_cont = ax.plot(time_array*obs_number/3600., significance, color=color, label=f"{md:.0f}m")

    def aimed_time(signi):
        return np.power(signi/sig_poisson.get_significance(sig_pe,bkg_pe).cpu().detach().numpy(), 2)

    time_flag = aimed_time(7)*obs_number/3600
    ax.plot([time_flag, time_flag], [0, 7], linestyle='--', color=color)

    time_at_ten_sigma[imd] = time_flag

ax.plot([0, time_at_ten_sigma[0]], [7, 7], linestyle='--', color='black')
ax.set_xlabel("Time [hour]")
ax.set_ylabel("Significance")
ax.set_xlim([0., 200.])
ax.set_ylim([0., 15])
#ax.set_xscale('log')

ax.legend()
#ax.legend(loc='upper right')

plt.show()




