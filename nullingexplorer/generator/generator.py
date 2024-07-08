import torch
import yaml

from nullingexplorer.model.amplitude import BaseAmplitude

class GenModel(BaseAmplitude):
    def __init__(self, config):
        super().__init__()