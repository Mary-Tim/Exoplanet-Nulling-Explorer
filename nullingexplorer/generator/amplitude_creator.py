import torch
import torch.nn as nn
import yaml
import numpy as np

from tensordict import TensorDict
from nullingexplorer.model.amplitude import BaseAmplitude
from nullingexplorer.utils import get_amplitude, get_instrument, get_spectrum, get_transmission, get_electronics
from nullingexplorer.utils import Configuration as cfg

class AmplitudeCreator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.load(config)

    def load(self, config):
        #print("Generate Amplitude")
        if isinstance(config, dict):
            self.config = config
        if isinstance(config, str):
            if config.endswith(('.yml', '.yaml',)):
                with open(config, mode='r', encoding='utf-8') as yaml_file:
                    self.config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

        # Config transmission map
        trans_config = 'DualChoppedDestructive'
        if config.get('TransmissionMap'):
            trans_config = config['TransmissionMap']
            if isinstance(trans_config, dict):
                trans_class = get_transmission(trans_config['Model'])
            else: trans_class = get_transmission(trans_config)
        
        # Regist amplitudes
        if config.get('Amplitude'):
            for name, val in config['Amplitude'].items():
                self.amplitude_register(name, val, trans_class)
                if isinstance(trans_config, dict):
                    if 'Buffers' in trans_config.keys():
                        self.buffer_setting(getattr(self, name).trans_map, config=trans_config['Buffers'])

        # Regist instrument
        inst_config = "MiYinBasicType"
        if config.get("Instrument"):
            inst_config = config["Instrument"]
            if isinstance(inst_config, dict):
                self.instrument = get_instrument(inst_config["Model"])()
                if 'Buffers' in inst_config.keys():
                    self.buffer_setting(self.instrument, config=inst_config['Buffers'])
            else:
                self.instrument = get_instrument(inst_config)()

        if config.get('Configuration'):
            for key, val in config['Configuration'].items():
                cfg.set_property(key, val)

        # Regist electronics background
        if config.get("Electronics"):
            elec_config = config['Electronics']
            if isinstance(elec_config, dict):
                self.electronics = get_electronics(elec_config['Model'])()
                if 'Buffers' in elec_config.keys():
                    self.buffer_setting(self.electronics, config=elec_config['Buffers'])
            else:
                self.electronics = get_electronics("UniformElectronics")()

    def forward(self, data):
        if hasattr(self, 'electronics'):
            return torch.sum(torch.stack([getattr(self, name)(data) for name in self.config['Amplitude']]), 0) * self.instrument(data) + self.electronics(data)
        else:
            return torch.sum(torch.stack([getattr(self, name)(data) for name in self.config['Amplitude']]), 0) * self.instrument(data)

    def amplitude_register(self, name, config, trans_class):
        self.__setattr__(name, get_amplitude(config['Model'])())
        amp = self.__getattr__(name)
        amp.trans_map = trans_class()
        if 'Spectrum' in config.keys():
            amp.spectrum = self.spectrum_register(config['Spectrum'])
        if 'Parameters' in config.keys():
            self.parameters_setting(amp, config=config['Parameters'])
        else:
            self.parameters_setting(amp, config=None)
        if 'Buffers' in config.keys():
            self.buffer_setting(amp, config=config['Buffers'])

    def spectrum_register(self, config):
        if isinstance(config, str):
            spectrum = get_spectrum(config)()
            return spectrum

        spectrum = get_spectrum(config['Model'])()
        if 'Parameters' in config['Spectrum'].keys():
            self.parameters_setting(spectrum, config['Spectrum']['Parameters'])
        else:
            self.parameters_setting(spectrum)

        return spectrum

    def buffer_setting(self, model: nn.Module, config: dict):
        for key, val in config.items():
            if hasattr(model, key):
                if isinstance(val, float):
                    getattr(model, key).data.fill_(val)
                elif isinstance(val, dict):
                    getattr(model, key).data.fill_(val['mean'])
            else:
                raise KeyError(f"Model {model} do not have buffer {key}")

    def parameters_setting(self, model, config=None):
        if not hasattr(model, 'boundary'):
            model.__setattr__("boundary", {})
        for key, param in model.named_parameters():
            if key.find('.') != -1:
                continue
            if key not in model.boundary.keys():
                model.boundary[key] = torch.tensor([-1.e6, 1.e6])
        if config is not None:
            for key, val in config.items():
                if isinstance(val, dict):
                    if not hasattr(model, key):
                        raise KeyError(f"Model {model} do not have parameter {key}")
                    getattr(model, key).data.fill_(val['mean'])
                    if 'min' in val.keys():
                        model.boundary[key][0] = float(val['min'])
                    if 'max' in val.keys():
                        model.boundary[key][1] = float(val['max'])
                    if 'fixed' in val.keys():
                        getattr(model, key).requires_grad = bool(val['fixed'])