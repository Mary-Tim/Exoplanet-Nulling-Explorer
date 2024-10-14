import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
import copy
plt.style.use(mplhep.style.LHCb2)

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from itertools import cycle
cycol = cycle('bgrcmk')

from nullingexplorer.utils import Constants as cons
from nullingexplorer.generator import AmplitudeCreator, ObservationCreator

amp_template = {
    'Amplitude':{
    },
    'Instrument': 'MiYinBasicType',
    'TransmissionMap': 'DualChoppedDifferential',
    'Configuration':{
        'distance': 10.,         # distance between Miyin and target [pc]
        'star_radius': cons._sun_radius,  # Star radius [kilometer]
        'star_temperature': 5773.,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 4.,
        'High': 18.5,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 1.,
        'IntegrationTime': 1.,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
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

amp_to_draw = {
    'Proxima_b':{
        'Model': 'PlanetPolarCoordinates',
        #'Model': 'PlanetWithReflection',
        'Spectrum': 'BinnedBlackBody',
        'Parameters':
        {
            'au':        {'mean': 1.},
            'polar':     {'mean': 0.},
        },
        'Spectrum': {
            'Model': 'RelativeBlackBodySpectrum',
            'Parameters':
            {
                'r_radius':         {'mean': 1.},
                'r_temperature':    {'mean': 1.},
            },
        },
    },
    'Stellar_Leak':{
        'Model': 'StarBlackBodyMatrix',
    },
    'Local_zodi':{
        'Model': 'LocalZodiacalDustMatrix',
    },
    'Exo_zodi':{
        "Model": 'ExoZodiacalDustMatrix',
    },
}

obs_creator = ObservationCreator()
obs_creator.load(obs_config)
data = obs_creator.generate()

def draw_line(ax, amp_name):
    amp_config = copy.deepcopy(amp_template)
    amp_config['Amplitude'][amp_name] = amp_to_draw[amp_name]

    amp_config['TransmissionMap'] = 'UnifiedTransmission'
    amp_unified = AmplitudeCreator(config=amp_config)
    data_without_trans = amp_unified(data)
    color = next(cycol)
    ax.plot(data['wl_mid'].cpu().detach().numpy()*1e6, data_without_trans.cpu().detach().numpy(), linestyle='-', color=color, label=f"{amp_name}")

    if amp_name != 'Proxima_b':
        amp_config['TransmissionMap'] = 'DualChoppedDestructive'
        amp_trans = AmplitudeCreator(config=amp_config)
        data_with_trans = amp_trans(data)
        ax.plot(data['wl_mid'].cpu().detach().numpy()*1e6, data_with_trans.cpu().detach().numpy(), linestyle=':', color=color)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    #draw_line(ax, 'Stellar_Leak')
    for key in amp_to_draw:
        draw_line(ax, key)
    ax.legend()
    ax.set_xlabel('Wavelength [$\mu m$]')
    ax.set_ylabel('Signal [$\\rm{{pe \\cdot s^{{-1}}}}$]')
    plt.show()