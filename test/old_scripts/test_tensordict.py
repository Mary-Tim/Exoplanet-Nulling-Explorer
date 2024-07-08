import numpy as np
import torch
from tensordict.prototype import tensorclass

torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda:0')

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    photon_electron: torch.Tensor


spectrum_range = np.array([5., 25.], dtype=np.float64) * 1e-6
bin_number = 5
phase_number = 10

phi = torch.tensor(np.repeat(np.linspace(0., 2*np.pi, phase_number), bin_number)).flatten()
center = torch.tensor([np.linspace(spectrum_range[0], spectrum_range[1], bin_number)]*phase_number).flatten()
width = torch.ones(phase_number*bin_number)*((spectrum_range[1]-spectrum_range[0])/bin_number)
values = torch.randn(phase_number*bin_number)

print(phi.size())
print(center.size())
print(width.size())
print(values.size())

data = MiYinData(phase=phi, wavelength=center, wl_width = width, photon_electron=values, batch_size=[phase_number*bin_number], device='cuda:0')

#print(data[99].phase)
for point in data:
    print(f"{point.phase:.03f}\t{point.wavelength:.03e}")
#print(data.select_label(50).photon_electron)

