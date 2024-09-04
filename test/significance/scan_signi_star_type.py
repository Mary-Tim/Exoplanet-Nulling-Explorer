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
from nullingexplorer.utils import Constants as cons

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

def habitable_zone(radius, temperature, distance):
    sun_radius = 695500
    sun_temperature = 5780
    habitable_parameters = np.array(
        [
            [1.7763,    1.4335e-4,  3.3954e-9,  -7.6364e-12,    -1.1950e-15],
            [0.3207,    5.4471e-5,  1.5275e-9,  -2.1709e-12,    -3.8282e-16]
        ],
        dtype=np.float32
    )

    relative_radius = radius / sun_radius
    relative_temperature = temperature / sun_temperature
    relative_lumi = relative_radius ** 2 * relative_temperature ** 4

    hz_temperature = temperature - sun_temperature

    s_eff_in = habitable_parameters[0, 0] + \
        habitable_parameters[0, 1] * hz_temperature + \
        habitable_parameters[0, 2] * hz_temperature ** 2 + \
        habitable_parameters[0, 3] * hz_temperature ** 3 + \
        habitable_parameters[0, 4] * hz_temperature ** 4

    s_eff_out = habitable_parameters[1, 0] + \
        habitable_parameters[1, 1] * hz_temperature + \
        habitable_parameters[1, 2] * hz_temperature ** 2 + \
        habitable_parameters[1, 3] * hz_temperature ** 3 + \
        habitable_parameters[1, 4] * hz_temperature ** 4

    #hz_in = np.sqrt(relative_lumi/s_eff_in) 
    #hz_out = np.sqrt(relative_lumi/s_eff_out) 
    hz_in = np.sqrt(relative_lumi/s_eff_in) * cons._au_to_meter / (distance * cons._pc_to_meter) * cons._radian_to_mas
    hz_out = np.sqrt(relative_lumi/s_eff_out) * cons._au_to_meter / (distance * cons._pc_to_meter) * cons._radian_to_mas

    return hz_in, hz_out

angular_separation = np.linspace(1., 300., 1000)

sig_poisson = PoissonSignificance()
sig_poisson.obs_config = obs_config

theta = 30. / 180. * np.pi
star_type = {
    'F': [6650., 1.275*695500],
    'G': [5650., 1.055*695500],
    'K': [4600., 0.83*695500],
}

fig, ax = plt.subplots()
for istar, st in tqdm(enumerate(star_type.keys()), total=len(star_type)):
    bkg_amp_config['Configuration']['star_temperature'] = star_type[st][0]
    bkg_amp_config['Configuration']['star_radius'] = star_type[st][1]
    bkg_pe = sig_poisson.gen_bkg_pe(bkg_amp_config)
    significance = np.zeros(len(angular_separation))
    for i, angular in enumerate(angular_separation):
        sig_amp_config['Amplitude']['earth']['Parameters']['ra']['mean'] = angular * np.cos(theta)
        sig_amp_config['Amplitude']['earth']['Parameters']['dec']['mean'] = angular * np.sin(theta)
        sig_pe = sig_poisson.gen_sig_pe(sig_amp_config)
        significance[i] = sig_poisson.get_significance(sig_pe, bkg_pe)

    color = next(cycol)
    trans_map_cont = ax.plot(angular_separation, significance, color=color, label=f"{st}-type")
    hz_in, hz_out = habitable_zone(star_type[st][1], star_type[st][0], 10.)

    print(f"Habitable zone: {hz_in:.2f} - {hz_out:.2f}")
    ax.fill_between([hz_in, hz_out], istar, istar+1, color=color, alpha=0.2)

ax.set_xlabel("Angular / mas")
ax.set_ylabel("Significance")
#ax.set_xlim([10., 200.])
#ax.set_ylim(bottom=0.1)
#ax.set_yscale('log')

ax.legend()

plt.show()




