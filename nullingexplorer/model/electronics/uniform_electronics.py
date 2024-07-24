import torch
import torch.nn as nn

from nullingexplorer.model.electronics import BaseElectronics

class UniformElectronics(BaseElectronics):
    def __init__(self):
        super(UniformElectronics, self).__init__()
        self.register_buffer('noise_rate', torch.tensor(1.0))

    def forward(self, data):
        return self.noise_rate * data['intg_time']