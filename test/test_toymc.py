import sys
sys.path.append('..')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.significance import ChiSquareSignificance

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
            'Value': 15.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'baseline': 10,         # nulling baseline [meter]
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

gen_amp_config = {
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

fit_amp_config = {
    'Amplitude':{
        'earth':{
            'Model': 'RelativePlanetBlackBody',
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                'r_radius':         {'mean': 1.e-5, 'min': 0., 'max': 5., 'fixed': False},
                'r_temperature':    {'mean': 1.e-5, 'min': 0., 'max': 5., 'fixed': False},
                'r_ra':            {'mean': 0., 'min': -2., 'max': 2., 'fixed': False},
                'r_dec':            {'mean': 0., 'min': -2., 'max': 2., 'fixed': False},
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

chisq_signi = ChiSquareSignificance(planet='earth', obs_config=obs_config, gen_amp_config=gen_amp_config, fit_amp_config=fit_amp_config)
chisq_signi.pseudoexperiments(number_of_toy_mc=3)