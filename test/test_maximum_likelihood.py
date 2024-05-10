import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.autograd as atg
from scipy.optimize import minimize, basinhopping, brute
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)
from tensordict.prototype import tensorclass

from nullingexplorer.amplitude import *
from nullingexplorer.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

phase_range=[0., np.pi*2]
phase_bins = 360
spectrum_range = np.array([5., 25.], dtype=np.float64) * 1e-6
spectrum_bins = 20
integral_time = 540.
earth_location=np.array([100./np.sqrt(2), 100./np.sqrt(2)]) / cons._radian_to_mac
earth_temperature = 285.

cfg.set_property('baseline', 15.)
cfg.set_property('ratio', 6.)
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor
    integral_time: torch.Tensor
    photon_electron: torch.Tensor

    def select_mod(self, mod):
        return self[self.mod==mod]

    def nparray(self, key):
        return getattr(self, key).cpu().detach().numpy()

phase = torch.tensor(np.repeat(np.linspace(phase_range[0], phase_range[1], phase_bins),spectrum_bins*2)).flatten()
wavelength = torch.tensor([np.repeat(np.linspace(spectrum_range[0], spectrum_range[1], spectrum_bins),2)]*phase_bins).flatten()
wl_width = torch.ones(phase_bins*spectrum_bins*2)*((spectrum_range[1]-spectrum_range[0])/spectrum_bins)
intg_time = torch.ones(phase_bins*spectrum_bins*2)*integral_time
mod = torch.tensor([np.array([1,-1])]*(phase_bins*spectrum_bins)).flatten()
photon_electron = torch.zeros(phase_bins*spectrum_bins*2)

data = MiYinData(phase=phase, wavelength=wavelength, wl_width=wl_width, mod=mod, integral_time=intg_time, photon_electron=photon_electron, batch_size=[phase_bins*spectrum_bins*2])

class Amplitude(BaseAmplitude):
    def __init__(self):
        super(Amplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.star = StarBlackBody()
        self.local_zodi = LocalZodiacalDust()
        self.exo_zodi = ExoZodiacalDust()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return self.earth(data) * self.instrument(data)
        #return (self.earth(data)+ self.mars(data) + self.venus(data)) * self.instrument(data)
        #return (self.earth(data) + self.star(data) + self.local_zodi(data) + self.exo_zodi(data)) * self.instrument(data)


amp = Amplitude()
for param in amp.named_parameters():
    print(param)
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(6371.e3)

data.photon_electron = torch.poisson(amp(data))
print(data.photon_electron)

## NLL function
#def negative_log_likelihood(data, model):
#    return torch.sum(model(data) - data.photon_electron * torch.log(model(data)))
#
## opt function
#def objective(params):
#    for val, par in zip(params, amp.parameters()):
#        par.data = torch.tensor(val)
#    NLL = negative_log_likelihood(data, amp)
#    grad = torch.stack(atg.grad(NLL, amp.parameters(), retain_graph=True, create_graph=True), dim=0)
#    return NLL.cpu().detach().numpy(), grad.cpu().detach().numpy()

class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        self.amp = Amplitude()
        self.dataset = None

    def set_data(self, data):
        self.dataset = data

    def forward(self):
        return torch.sum(self.amp(self.dataset) - self.dataset.photon_electron * torch.log(self.amp(self.dataset)))

    def objective(self, params):
        if self.dataset == None:
            raise ValueError('Dataset should be initialized before call the NLL!')
        for val, par in zip(params, self.amp.parameters()):
            par.data = torch.tensor(val)
        NLL = self.forward()
        grad = torch.stack(atg.grad(NLL, self.amp.parameters(), retain_graph=True, create_graph=True), dim=0)
        return NLL.cpu().detach().numpy(), grad.cpu().detach().numpy()

# 初始化NLL
NLL = NegativeLogLikelihood()
NLL.set_data(data)
# 定义参数的范围约束
initial_val = [param.item() for param in amp.parameters()]
bounds = [(6000000, 7000000), (250., 300.), (-80./cons._radian_to_mac, 80./cons._radian_to_mac), (-80./cons._radian_to_mac, 80./cons._radian_to_mac)]
NLL.amp.earth.ra.data = torch.tensor(65./cons._radian_to_mac)
NLL.amp.earth.dec.data = torch.tensor(75./cons._radian_to_mac)
# fix temperature and 

#objective = lambda params : NLL.objective(params)[0]
#finish_func = lambda func, x0, args: minimize(NLL.objective, x0=x0, bounds=bounds, method='L-BFGS-B', jac=True, options={'disp': True})

# 计时
import time
start_time = time.time()
#result = minimize(NLL.objective, x0=initial_val, bounds=bounds, method='L-BFGS-B', jac=True)
#print(f"Initial NLL: {result.fun}")
#for i in range(10000):
#    init_val = [np.random.uniform(low=val[0], high=val[1]) for val in bounds]
#    this_result = minimize(NLL.objective, x0=init_val, bounds=bounds, method='L-BFGS-B', jac=True)
#    print(f"New guess {i}: {init_val}, NLL: {this_result.fun}")
#    if this_result.fun < result.fun:
#        result = this_result
#        print(f'find new minimum at iter {i}: NLL: {result.fun}')


#result = minimize(NLL.objective, x0=initial_val, bounds=bounds, method='L-BFGS-B', jac=True)
result = basinhopping(NLL.objective, 
                      x0=initial_val, 
                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds': bounds, 'jac': True, 
                                        'options': {'maxcor': 50, 'ftol': 1e-15, 'maxiter': 10000}}, 
                      disp=True)
#x0, fval, grid, Jout = brute(objective, ranges=bounds, Ns=5, full_output=True, finish=finish_func)
#
#cov_matrix = result.hess_inv.todense()
cov_matrix = result.lowest_optimization_result.hess_inv.todense()

end_time = time.time()
print(f"最小化用时：{(end_time-start_time)*1e3} ms")

print(f"HESSE Matrix:\n{cov_matrix}")

# 打印拟合结果
print("拟合结果:")
print(f"radius:\t{amp.earth.radius.item():6.03f} +/- {np.sqrt(cov_matrix[0,0]):6.03f}")
print(f"temperature:\t{amp.earth.temperature.item():6.03f} +/- {np.sqrt(cov_matrix[1,1]):6.03f}")
print(f"ra:\t{amp.earth.ra.item()*cons._radian_to_mac:6.03f} +/- {np.sqrt(cov_matrix[2,2])*cons._radian_to_mac:6.03f}")
print(f"dec:\t{amp.earth.dec.item()*cons._radian_to_mac:6.03f} +/- {np.sqrt(cov_matrix[3,3])*cons._radian_to_mac:6.03f}")
