import sys
sys.path.append('..')

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

from itertools import cycle
cycol = cycle('bgrcmk')

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
        'IntegrationTime': 1,  # unit: second
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
    'TransmissionMap': {'Model': 'DualChoppedDifferential'},
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
    'Electronics': {
        'Model': 'UniformElectronics',
        'Buffers': {
            'noise_rate': 1.0,
            },
        },
    'Configuration':{
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}
max_time = 1600.
intg_time = torch.linspace(0., max_time, 1000)
elec_noise_rate = np.logspace(0, 3, 4)
obs_number = obs_config['Observation']['ObsNumber']

sig_poisson = PoissonSignificance()
sig_poisson.obs_config = obs_config

sig_poisson.sig_amp_config = sig_amp_config
sig_pe = sig_poisson.gen_sig_pe()

time_at_ten_sigma = np.zeros(len(elec_noise_rate))

fig, ax = plt.subplots()
def sig_elec_noise(elec_rate):
    bkg_amp_config['Electronics']['Buffers']['noise_rate'] = elec_rate
    sig_poisson.bkg_amp_config = bkg_amp_config
    bkg_pe = sig_poisson.gen_bkg_pe()
    color = next(cycol)
    def sig_point(time):
        return sig_poisson.get_significance(sig_pe*time, bkg_pe*time)
    lineshape_noise = torch.vmap(sig_point)(intg_time)
    ax.plot(intg_time.cpu().detach().numpy()*obs_number/3600, lineshape_noise.cpu().detach().numpy(), color=color, label=f"$10^{np.log10(elec_rate):.0f}$")

    def aimed_time(signi):
        return np.power(signi/sig_point(1.0).cpu().detach().numpy(), 2)

    time_flag = aimed_time(10)*obs_number/3600
    ax.plot([time_flag, time_flag], [0, 10], linestyle='--', color=color)

    return time_flag

for i, elec_rate in tqdm(enumerate(elec_noise_rate), total=len(elec_noise_rate)):
    time_at_ten_sigma[i] = sig_elec_noise(elec_rate)

ax.plot([0, time_at_ten_sigma[-1]], [10, 10], linestyle='--', color='black')
ax.set_xlabel("Time [hour]")
ax.set_ylabel("Significance [$\\rm{{\\sigma}}$]")
ax.set_xlim([0., max_time*obs_number/3600.])
ax.set_ylim([0., 12.])
ax.legend(fontsize=30, loc='lower center')

for i in range(len(elec_noise_rate)):
    print(f"elec: {elec_noise_rate[i]}, time: {time_at_ten_sigma[i]:.2f} / hour")

plt.show()




