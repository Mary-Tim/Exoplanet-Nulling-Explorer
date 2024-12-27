import torch
import yaml
import numpy as np
from datetime import datetime

from tensordict import TensorDict
from nullingexplorer.utils import Configuration as cfg

class ObservationCreator():
    def __init__(self):
        self.__config = None
        self.__spec_gen = SpectrumGenerator()
        self.__obs_gen = ObsGenerator()
        self.__tf_gen = TimeFlagGenerator()
        self.obs_num = None
        self.spec_num = None
        self.mod_num = None

    def load(self, config):
        '''
        Create the Observation plan of MiYin dataset
        
        : param config: <str or dict> configuration of observation plan
            str: Path of a yaml file
            dict: Config dict

        An example of config:
        {
            'Spectrum':
            {
                'Type': 'Equal',
                'BinNumber': 100,
                'Low': 5.,
                'High': 25.,                # unit: micrometer
            },
            'Observation':
            {
                'ObsNumber': 360,
                'IntegrationTime': 300,     # unit: second
                'ObsMode': [1, -1]          # For chopped nulling
                'Phase'
                {
                    'Start' : 0.,
                    'Stop': 360.,           # unit: degree
                },
                'Baseline':
                {
                    'Type': 'Constant',
                    'Value': 100.,          # unit: meter
                },
                'TimeFlag':
                {
                    'StartEpoch': '2023-01-01 00:00:00',
                    'ControlTime': 1000,    # unit: second
                },
            },
        }
        '''
        if isinstance(config, dict):
            self.__config = config
        if isinstance(config, str):
            if config.endswith(('.yml', '.yaml',)):
                with open(config, mode='r', encoding='utf-8') as yaml_file:
                    self.__config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    def generate(self) -> TensorDict:
        if not self.__config:
            raise ValueError('Config is not loaded.')
        
        if self.__config.get('Configuration'):
            for key, val in self.__config['Configuration'].items():
                cfg.set_property(key, val)

        obs_num = self.__obs_gen.call_observation(self.__config['Observation'])
        spec_num = self.__spec_gen.call_spectrum(self.__config['Spectrum'])
        #tf_num = self.__tf_gen.call_timeflag(self.__config['TimeFlag'], self.__obs_gen)
        start_time = self.__tf_gen.call_timeflag(self.__config.get('TimeFlag'), self.__obs_gen)
        mod_num = len(self.__obs_gen.mod)
        batch_size = obs_num*spec_num*mod_num

        data = TensorDict({
            'phase': torch.tensor(np.repeat(self.__obs_gen.phase, spec_num*mod_num)).flatten(),
            'baseline': torch.tensor(np.repeat(self.__obs_gen.baseline, spec_num*mod_num)).flatten(),
            'start_time': torch.repeat_interleave(start_time, repeats=spec_num*mod_num),
            'intg_time': torch.ones(batch_size)*self.__obs_gen.intg_time,
            'wl_lo': torch.tensor(np.array([np.repeat(self.__spec_gen.wl_lo, mod_num)]*obs_num)).flatten(),
            'wl_hi': torch.tensor(np.array([np.repeat(self.__spec_gen.wl_hi, mod_num)]*obs_num)).flatten(),
            'wl_mid': torch.tensor(np.array([np.repeat(self.__spec_gen.wl_mid, mod_num)]*obs_num)).flatten(),
            'mod': torch.tensor(np.array(self.__obs_gen.mod*obs_num*spec_num)).flatten(),
            'photon_electron': torch.zeros(batch_size),
            'pe_uncertainty': torch.zeros(batch_size),
        }, batch_size=[batch_size])

        self.obs_num = obs_num
        self.spec_num = spec_num
        self.mod_num = mod_num

        return data

class SpectrumGenerator():
    def __init__(self) -> None:
        self.spec_num = None
        self.wl_lo = None
        self.wl_hi = None
        self.wl_mid = None
        self.__config = None

    def call_spectrum(self, config):
        self.__config = config
        if self.__config['Type'] == 'Equal':
            self.equal_spectrum()
        elif config['Type'] == 'Resolution':
            self.res_spectrum()

        self.wl_lo = self.wl_lo * 1e-6
        self.wl_hi = self.wl_hi * 1e-6
        self.wl_mid = (self.wl_lo + self.wl_hi) / 2.

        return self.spec_num
        
    def equal_spectrum(self):
        self.spec_num = int(self.__config["BinNumber"])
        wl_bins = np.linspace(self.__config['Low'], self.__config['High'], self.spec_num+1)
        self.wl_lo = wl_bins[:-1]
        self.wl_hi = wl_bins[1:]

    def res_spectrum(self):
        max_length = 10000
        self.spec_num = 0
        self.R_val = float(self.__config['R'])
        if self.R_val <= 0.5:
            raise ValueError('Resolution must be greater than 0.5.')
        self.wl_lo = np.zeros(max_length)
        self.wl_hi = np.zeros(max_length)
        lo_end = self.__config['Low']
        hi_end = self.__config['High']

        def next_hi(low):
            width = low / (self.R_val-0.5)
            return low + width
        lo = lo_end
        while(1):
            hi = next_hi(lo)
            self.wl_lo[self.spec_num] = lo
            self.wl_hi[self.spec_num] = hi
            self.spec_num += 1
            lo = hi
            if hi > hi_end:
                break
        if self.spec_num > max_length:
            raise ValueError('Wavelength list overflow. Please enlarge the max_length.')

        self.wl_lo = self.wl_lo[:self.spec_num]
        self.wl_hi = self.wl_hi[:self.spec_num]
        
class ObsGenerator():
    def __init__(self) -> None:
        self.obs_num = None
        self.phase = None
        self.baseline = None
        self.intg_time = None
        self.mod = None
        self.distribution = {
            "Linear": self.linear,
            "Log": self.log,
            "Exp": self.exp,
        }

    def call_observation(self, config):
        self.obs_num = int(config['ObsNumber'])
        self.phase = self.linear(config['Phase']['Start'], config['Phase']['Stop'], self.obs_num) / 180. * np.pi
        self.intg_time = config['IntegrationTime']
        self.mod = config['ObsMode']

        bl_config = config['Baseline']
        if bl_config['Type'] == 'Constant':
            self.baseline = np.ones(self.obs_num) * bl_config['Value']
        elif bl_config['Type'] in self.distribution.keys():
            self.baseline = self.distribution[bl_config['Type']](bl_config['Low'], bl_config['High'], self.obs_num)
        else:
            raise ValueError('Baseline type is not supported.')

        return self.obs_num

    def linear(self, low, high, num):
        return np.linspace(low, high, num)

    def log(self, low, high, num):
        return np.log(np.linspace(1., np.e, num)) * (high-low) + low

    def exp(self, low, high, num):
        return (np.exp(np.linspace(0, np.log(2), num))-1.) * (high-low) + low

class TimeFlagGenerator():
    def __init__(self) -> None:
        self.start_time = None

    def call_timeflag(self, config=None, obs_gen:ObsGenerator=None):
        integral_time = obs_gen.intg_time
        observe_number = obs_gen.obs_num
        if config is None:
            self.start_time = torch.zeros(observe_number)
        else:
            start_epoch_str = config['StartEpoch']
            control_time = config['ControlTime']
            self.start_epoch = datetime.strptime(start_epoch_str, '%Y-%m-%d %H:%M:%S').timestamp()
            self.start_time = self.start_epoch + torch.arange(observe_number) * (integral_time + control_time)
        
        return self.start_time