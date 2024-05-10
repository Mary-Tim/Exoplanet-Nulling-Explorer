import sys
sys.path.append('..')

import torch
import torch.nn as nn
from tensordict.prototype import tensorclass

from nullingexplorer.utils import get_amplitude
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer import *

torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda:0')

import numpy as np
import matplotlib.pyplot as plt

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor

cfg.set_property('distance', 10.)
cfg.set_property('baseline', 15.)
cfg.set_property('target_latitude', 45.)
cfg.set_property('target_longitude', 90.)

spectrum_range = np.array([5., 25.], dtype=np.float64) * 1e-6
bin_number = 100
phase_number = 1

phi = torch.tensor(np.repeat(np.linspace(0., 2*np.pi, phase_number), bin_number)).flatten()
center = torch.tensor([np.linspace(spectrum_range[0], spectrum_range[1], bin_number)]*phase_number).flatten()
width = torch.ones(phase_number*bin_number)*((spectrum_range[1]-spectrum_range[0])/bin_number)
mod = torch.ones(phase_number*bin_number)

data1 = MiYinData(phase=phi, wavelength=center, wl_width = width, mod = mod, batch_size=[phase_number*bin_number], device='cuda:0')
data2 = MiYinData(phase=phi, wavelength=center, wl_width = width, mod = -mod, batch_size=[phase_number*bin_number], device='cuda:0')

model = get_amplitude('LocalZodiacalDust')()
data = data1

result = model(data).cpu().detach().numpy()
print(result)


plt.plot(center.cpu().detach().numpy(), model(data).cpu().detach().numpy(), 'r-', label='fit')

plt.show()