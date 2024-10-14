import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
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
        'Type': 'Resolution',
        'R': 20,
        'Low': 0.9,
        'High': 1.7,        # unit: micrometer
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
    'Instrument': 'MiYinBasicType',
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
    'Instrument': 'MiYinBasicType',
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

temperature_range = np.array([200., 800.])
radius_range = np.array([3000., 8000.])
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
    sig_amp_config['Amplitude']['earth']['Spectrum']['Parameters']['r_temperature']['mean'] = temp / 285.
    sig_amp_config['Amplitude']['earth']['Spectrum']['Parameters']['r_radius']['mean'] = radius / 6371.
    sig_poisson.sig_amp_config = sig_amp_config
    sig_pe = sig_poisson.gen_sig_pe_single()
    return sig_poisson.get_significance_single(sig_pe, bkg_pe)

significance = np.zeros(len(temperature_array), dtype=np.float64)

for i, temp, radius in tqdm(zip(range(len(temperature_array)), temperature_array, radius_array), total=len(temperature_array)):
    significance[i] = sig_point(temp, radius)

print(significance)
fig, ax = plt.subplots()
#levels = np.logspace(np.log10(np.min(significance)*0.99), np.log10(1.01*np.max(significance)), 1000)
levels = np.geomspace(0.001, 1.01*np.max(significance), 1000)
#locator = ticker.LogLocator(base=10.0)
significance = significance.reshape(scan_num, scan_num)
#trans_map_cont = ax.contourf(TEMP, RADI, significance, levels=levels, locator=locator, cmap = plt.get_cmap("gist_rainbow"))
#trans_map_cont = ax.contourf(TEMP, RADI, significance, cmap = plt.get_cmap("gist_rainbow"), norm='log')
trans_map_cont = ax.contourf(TEMP, RADI, significance, levels=levels, cmap = plt.get_cmap("gist_rainbow"), norm='log')
trans_7_sigma = ax.contour(TEMP, RADI, significance, levels=[7], colors='white', linewidths=2)  # , label="7$\,\sigma$"
plt.clabel(trans_7_sigma, inline=True, fontsize=35, fmt='7$\,\\rm{{\sigma}}$')
#trans_map_cont = ax.contourf(TEMP, RADI, significance.reshape(scan_num, scan_num), levels=levels, cmap = plt.get_cmap("bwr"), norm='log')

ax.set_xlabel("Temperature / K")
ax.set_ylabel("Radius / km")

cbar = fig.colorbar(trans_map_cont, format="%.2f")

## Mark planets
#planets = {
#    'Earth': [285., 6371., 'P'],
#    'Mars':  [210., 3389., 'o'],
#    'Venus': [737., 6051., 'v'],
#}
#
#signi = []
#for val in planets.values():
#    signi.append(sig_point(val[0], val[1]))
#    #signi.append(interpn(points=(temperature_line, radius_line), values=significance.reshape(scan_num, scan_num), xi=np.array([val[0], val[1]])))
#
#print(signi)
#
#for i, name, val in zip(range(len(planets)), planets.keys(), planets.values()):
#    ax.scatter(val[0], val[1], s=200, marker=val[2], color='black', label=name)
#    plt.annotate(text=f"{signi[i]:.2f}$\,\\rm{{\sigma}}$", xy=(val[0], val[1]), xytext=(val[0]-45, val[1]+150), fontsize=25, color='black')
#
#ax.legend(fontsize=30, loc='lower right')

plt.show()




