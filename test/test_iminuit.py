import sys
sys.path.append('..')

import torch

from nullingexplorer.generator import AmplitudeCreator, ObservationCreator
from nullingexplorer.utils import Constants as cons
from nullingexplorer.io import DataHandler
from nullingexplorer.fitter import MinuitFitter

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

gen_amp_config = {
    'Amplitude':{
        'earth':{
            'Model': 'PlanetBlackBody',
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                'radius':         {'mean': 6371.e3},
                'temperature':    {'mean': 285.},
                #'ra':            {'mean': 50.},
                #'dec':            {'mean': 100.},
                'ra':            {'mean': 62.5},
                'dec':            {'mean': 78.1},
            },
        },
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
            #'Model': 'RelativePlanetBlackBody',
            'Model': 'RelativePolarPlanetBlackBody',
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                'r_radius':         {'mean': 1.e-5, 'min': 0., 'max': 5., 'fixed': False},
                'r_temperature':    {'mean': 1.e-5, 'min': 0., 'max': 5., 'fixed': False},
                #'r_ra':             {'mean': 0., 'min': -2., 'max': 2., 'fixed': False},
                #'r_dec':            {'mean': 0., 'min': -2., 'max': 2., 'fixed': False},
                'r_angular':        {'mean': 1., 'min': 0.1, 'max': 3., 'fixed': False},
                'r_polar':          {'mean': 0., 'min': 0., 'max': 2.*torch.pi, 'fixed': False},
            },
        },
    },
    'Instrument': 'MiYinBasicType',
    #'TransmissionMap': 'DualChoppedDifferential',
    'TransmissionMap': 'PolarDualChoppedDifferential',
    'Configuration':{
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

def main():
    # Set device during generation
    torch.set_default_device('cuda:0')
    torch.set_default_dtype(torch.float64)
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Observation Config
    obs_creator = ObservationCreator()
    obs_creator.load(obs_config)
    data = obs_creator.generate()

    # Simulation
    print("Generating MC ......")
    gen_amp = AmplitudeCreator(config=gen_amp_config)
    data['photon_electron'] = torch.poisson(gen_amp(data))
    data_handler = DataHandler(data)
    diff_data = data_handler.diff_data(obs_creator)

    # Estimation
    print("Fitting ......")
    fit_model = AmplitudeCreator(config=fit_amp_config)
    #fit_model_compile = torch.compile(fit_model) # torch.compile doesn't support AMD Radeon VII (gfx906)
    fitter = MinuitFitter(fit_model, diff_data, multi_gpu=True)
    fitter.search_planet('earth', draw=True, std_err=True, show=True, random_number=4000, position_name=['earth.r_polar', 'earth.r_angular'], polar=True)

    print(f"NLL without earth: {fitter.NLL.call_nll_nosig()}")

if __name__ == '__main__':
    main()