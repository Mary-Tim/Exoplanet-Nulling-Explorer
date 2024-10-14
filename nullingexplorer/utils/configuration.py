import torch
import numpy as np

class Configuration():
    # Default values
    _data_property = {
        # Target parameters
        'distance': 10,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
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
        'trans_map': 'DualChoppedDestructive', # Nulling transmission map [dimensionless]
        'fov_scale': 1.         # Scale of FoV
    } # {key: torch.Tensor}
    for key, val in _data_property.items():
        if type(val) == str:
            continue
        elif type(val) != torch.Tensor:
            _data_property[key] = torch.tensor(float(val))

    @classmethod
    def set_property(cls, key: str, val):
        if isinstance(val, (float, list, np.ndarray, np.floating)):
            cls._data_property[key] = torch.tensor(val)
        elif isinstance(val, torch.Tensor):
            cls._data_property[key] = val
        elif isinstance(val, int):
            cls._data_property[key] = torch.tensor(float(val))
        elif isinstance(val, str):
            cls._data_property[key] = val
        else:
            raise TypeError(f"Configure: Type {type(val)} is not an available type of data property!")

    @classmethod
    def get_property(cls, key: str) -> torch.Tensor:
        if key not in cls._data_property.keys():
            raise ValueError(f'Configure: Data property {key} not found!')
        if type(cls._data_property[key]) is str:
            return cls._data_property[key]
        return cls._data_property[key].detach().clone()

    @classmethod
    def print_available_data_property(cls):
        print('All available data property:')
        for name, val in cls._data_property.items():
            print(f"Property {name}:\t{val}")

    @classmethod
    def init_obs_config(cls, data):
        pass