import sys
sys.path.append('..')

import torch
import torch.nn as nn
from tensordict.prototype import tensorclass
from nullingexplorer.utils import get_spectrum
from nullingexplorer import *

torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda:0')

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor

import numpy as np
import matplotlib.pyplot as plt

#model = BlackBody()
model = get_spectrum('BinnedBlackBody')()
#model = get_spectrum('TorchQuadBlackBody')()
temperature = torch.tensor(150.)
spectrum_range = np.array([5., 25.], dtype=np.float64) * 1e-6
bin_number = 100

center = torch.tensor(np.linspace(spectrum_range[0], spectrum_range[1], bin_number))
width = torch.ones(bin_number)*((spectrum_range[1]-spectrum_range[0])/bin_number)
phi = torch.zeros(bin_number)
#data = {'wavelength': center, 'wl_width': width}

data = MiYinData(phase=phi, wavelength=center, wl_width=width, batch_size=[bin_number])

result = model(temperature, data)
print(result)

plt.plot(center.cpu().detach().numpy(), model(temperature, data).cpu().detach().numpy(), 'r-', label='fit')
temperature = torch.tensor(200.)
plt.plot(center.cpu().detach().numpy(), model(temperature, data).cpu().detach().numpy(), 'b-', label='fit')

plt.show()