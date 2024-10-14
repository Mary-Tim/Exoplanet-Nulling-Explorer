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
mkcol = cycle('ov^*sX')

from nullingexplorer.utils import Constants as cons
from nullingexplorer.generator import AmplitudeCreator, ObservationCreator

amp_template = {
    'Amplitude':{
    },
    'Instrument': 'MiYinTwoMirror',
    #'TransmissionMap': 'DualChoppedDifferential',
    #'TransmissionMap': 'DualChoppedDestructive',
    'TransmissionMap': 'SingleBracewell',
    'Configuration':{
        #'distance': 10.,         # distance between Miyin and target [pc]
        'distance': 4.2 * cons._light_year_to_meter / cons._pc_to_meter,         # distance between Miyin and target [pc]
        #'star_radius': 0.0002*cons._sun_radius,  # Star radius [kilometer]
        'star_radius': 0.1542*cons._sun_radius,  # Star radius [kilometer]
        'star_temperature': 2992.,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': -62.67,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 0.5,
        'High': 3.,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 1./360.,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 3.,  # unit: meter
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

amp_to_draw = {
    #'Proxima_b_em':{
    #    'Model': 'PlanetPolarCoordinates',
    #    #'Model': 'PlanetWithReflection',
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
    'Proxima_b':{
        #'Model': 'PlanetPolarCoordinates',
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

trans_type = {'UnifiedTransmission': '-', 'DualChoppedDestructive': ':'}

obs_creator = ObservationCreator()

def draw_line(ax, bl):
    amp_name = 'Stellar_Leak'
    amp_config = copy.deepcopy(amp_template)
    amp_config['Amplitude'][amp_name] = amp_to_draw[amp_name]

    obs_config['Observation']['Baseline']['Value'] = bl
    obs_creator.load(obs_config)
    data = obs_creator.generate()
    data_0 = data.reshape(obs_creator.obs_num, obs_creator.spec_num)
    wl_width = (data_0[0]['wl_hi'] - data_0[0]['wl_lo']) * 1e6
    color = next(cycol)

    if bl == 3:
        amp_config['TransmissionMap'] = 'UnifiedTransmission'
        amp_unified = AmplitudeCreator(config=amp_config)
        data_without_trans = amp_unified(data)
        data_without_trans = data_without_trans.reshape(obs_creator.obs_num, obs_creator.spec_num)
        data_without_trans = torch.sum(data_without_trans, dim=0)
        ax.plot(data_0[0]['wl_mid'].cpu().detach().numpy()*1e6, (data_without_trans/wl_width).cpu().detach().numpy(), linestyle='-', color=color, label="Origin")
        color = next(cycol)

    amp_config['TransmissionMap'] = 'SingleBracewell'
    amp_trans = AmplitudeCreator(config=amp_config)
    data_with_trans = amp_trans(data)
    data_with_trans = data_with_trans.reshape(obs_creator.obs_num, obs_creator.spec_num)
    data_with_trans = torch.sum(data_with_trans, dim=0)
    ax.plot(data_0[0]['wl_mid'].cpu().detach().numpy()*1e6, (data_with_trans/wl_width).cpu().detach().numpy(), linestyle=':', color=color, label=f"{bl:d}m")

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    #draw_line(ax, 'Stellar_Leak')
    baseline = np.array([3,5,10,20])
    for bl in baseline:
        draw_line(ax, bl)
    ax.set_xlabel('Wavelength [$\\rm{{\mu m}}$]')
    ax.set_ylabel('Signal [$\\rm{{pe \\cdot s^{{-1}} \\cdot \mu m^{{-1}}}}$]')
    ax.grid(True)
    ax.legend()
    plt.show()