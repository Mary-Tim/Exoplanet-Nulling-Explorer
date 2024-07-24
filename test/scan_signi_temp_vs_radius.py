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

from nullingexplorer.generator import PoissonSignificance

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
        'IntegrationTime': 200,  # unit: second
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
    'Instrument': 'MiYinBasicType',
    'TransmissionMap': 'DualChoppedDifferential',
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

temperature_range = np.array([100, 800])
radius_range = np.array([6371*0.48, 6371*1.1])
scan_num = 100

temperature_line = np.linspace(temperature_range[0], temperature_range[1], scan_num)
radius_line = np.linspace(radius_range[0], radius_range[1], scan_num)

TEMP, RADI = np.meshgrid(temperature_line, radius_line)

temperature_array = TEMP.flatten()
radius_array = RADI.flatten()

sig_poisson = PoissonSignificance()
sig_poisson.obs_config = obs_config
sig_poisson.bkg_amp_config = bkg_amp_config

bkg_pe = sig_poisson.gen_bkg_pe()

def sig_point(temp, radius):
    sig_amp_config['Amplitude']['earth']['Parameters']['temperature']['mean'] = temp
    sig_amp_config['Amplitude']['earth']['Parameters']['radius']['mean'] = radius*1e3
    sig_poisson.sig_amp_config = sig_amp_config
    sig_pe = sig_poisson.gen_sig_pe()
    return sig_poisson.get_significance(sig_pe, bkg_pe)

significance = np.zeros(len(temperature_array))

for i, temp, radius in tqdm(zip(range(len(temperature_array)), temperature_array, radius_array), total=len(temperature_array)):
    #print(temp, radius)
    significance[i] = sig_point(temp, radius)

fig, ax = plt.subplots()
levels = np.geomspace(0.001, 1.01*np.max(significance), 1000)
trans_map_cont = ax.contourf(TEMP, RADI, significance.reshape(scan_num, scan_num), levels=levels, cmap = plt.get_cmap("bwr"), norm='log')

ax.set_xlabel("Temperature / Kelvin")
ax.set_ylabel("Radius / 1e3 km")

cbar = fig.colorbar(trans_map_cont, format="%.2f")

# Mark planets
planets = {
    'Earth': [285., 6371., 'P'],
    'Mars':  [210., 3389., 'o'],
    'Venus': [737., 6051., 'v'],
}

signi = []
for val in planets.values():
    signi.append(sig_point(val[0], val[1]))
    #signi.append(interpn(points=(temperature_line, radius_line), values=significance.reshape(scan_num, scan_num), xi=np.array([val[0], val[1]])))

print(signi)

for i, name, val in zip(range(len(planets)), planets.keys(), planets.values()):
    ax.scatter(val[0], val[1], s=200, marker=val[2], color='black', label=name)
    plt.annotate(text=f"{signi[i]:.2f}$\,\sigma$", xy=(val[0], val[1]), xytext=(val[0]-40, val[1]-250), fontsize=25, color='black')

ax.legend(fontsize=30, loc='lower right')

plt.show()




