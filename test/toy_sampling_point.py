import sys
sys.path.append('..')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

import numpy as np


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
        'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 2,   # Diameter of MiYin primary mirror [meter]
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
        'star_temperature': 5778,   # Star temperature [Kelvin]
        'target_longitude': 135.,     # Ecliptic longitude [degree]
        'target_latitude': 45.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

fit_amp_config = {
    'Amplitude':{
        'earth':{
            'Model': 'RelativePolarPlanetBlackBody',
            'Spectrum': 'BinnedBlackBody',
            'Parameters':
            {
                #'r_radius':         {'mean': 1.e-5, 'min': 0.2, 'max': 3., 'fixed': False},
                #'r_temperature':    {'mean': 1.e-5, 'min': 0.2, 'max': 3., 'fixed': False},
                #'r_angular':        {'mean': 1., 'min': 0.2, 'max': 2., 'fixed': False},
                #'r_polar':          {'mean': 0., 'min': 0., 'max': 2.*torch.pi, 'fixed': False},
                'r_radius':         {'mean': 1.e-5, 'min': 0., 'max': 5., 'fixed': False},
                'r_temperature':    {'mean': 1.e-5, 'min': 0., 'max': 5., 'fixed': False},
                'r_angular':        {'mean': 1., 'min': 0.2, 'max': 2., 'fixed': False},
                'r_polar':          {'mean': 0., 'min': 0., 'max': 2.*torch.pi, 'fixed': False},
            },
        },
    },
    'Instrument': 'MiYinBasicType',
    'TransmissionMap': 'PolarDualChoppedDifferential',
    'Configuration':{
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5778,   # Star temperature [Kelvin]
        'target_longitude': 135.,     # Ecliptic longitude [degree]
        'target_latitude': 45.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

def main():
    torch.multiprocessing.set_start_method('spawn') # MUST BE SETTED to enable the JoinableQueue in multiprocessing

    from nullingexplorer.significance import ToyMonteCarlo

    integral_time = 200000
    obs_num = 360
    num_of_toy = 1000
    #mas_range = [50., 150.]
    mas = 100.

    obs_config['Observation']['ObsNumber'] = obs_num
    obs_config['Observation']['IntegrationTime'] = integral_time/float(obs_num)

    toy_mc = ToyMonteCarlo(multi_gpu=True)
    try: 
        for i in range(num_of_toy):
            #mas = np.random.uniform(mas_range[0], mas_range[1])
            theta = np.random.uniform(0., 2*np.pi)
            ra = np.cos(theta) * mas
            dec = np.sin(theta) * mas
            gen_amp_config['Amplitude']['earth']['Parameters']['ra']['mean']  = ra
            gen_amp_config['Amplitude']['earth']['Parameters']['dec']['mean'] = dec

            print('*'*30)
            print(f"Processing toy {i+1} ......")
            print(f"Truth position -- angular: {mas:3f},\tpolar: {theta:3f}")
            print('*'*30)
            toy_mc.do_a_toy(gen_amp_config, fit_amp_config, obs_config, random_fit_number=100, save_toy_result=True, position_name=['earth.r_polar', 'earth.r_angular'], polar=True)
    except:
        toy_mc.save_all()
    else:
        toy_mc.save_all()

if __name__ == '__main__':
    main()
