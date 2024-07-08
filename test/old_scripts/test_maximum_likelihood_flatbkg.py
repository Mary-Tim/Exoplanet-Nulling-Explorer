import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from scipy.interpolate import griddata
from tqdm import tqdm

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
spectrum_bins = 30
integral_time = 540.
earth_location=np.array([100./np.sqrt(2), 100./np.sqrt(2)]) / cons._radian_to_mas
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

    def select_data(self, key, val):
        return self[getattr(self, key)==val]

    def select_wavelength(self, wl):
        return self[self.wavelength==wl]

    def nparray(self, key):
        return getattr(self, key).cpu().detach().numpy()

    def get_bins(self, key):
        return torch.unique(getattr(self, key))

    def get_bin_numbers(self, key):
        return torch.tensor(len(torch.unique(getattr(self, key))))

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
        #return self.earth(data) * self.instrument(data)
        #return (self.earth(data)+ self.mars(data) + self.venus(data)) * self.instrument(data)
        return (self.earth(data) + self.star(data) + self.local_zodi(data) + self.exo_zodi(data)) * self.instrument(data)


amp = Amplitude()
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(6371.e3)

data.photon_electron = torch.poisson(amp(data))

class ShotNoise(BaseAmplitude):
    def __init__(self):
        super(ShotNoise, self).__init__()
        self.register_buffer('spectrum_num', torch.tensor(0))
        self.register_buffer('phase_num', torch.tensor(0))
        self.register_buffer('mod_num', torch.tensor(0))
        self.register_buffer('wl_width', torch.tensor([]))

        self.spectrum_num.data = torch.tensor(spectrum_bins)

        for i in range(self.spectrum_num):
            self.register_parameter(f'bkg{i}', torch.tensor(1.))

    def initialize(self, data):
        spectrum_num = data.get_bin_numbers("wavelength")
        wl_width_num = data.get_bin_numbers("wl_width")
        if wl_width_num == 1:
            self.wl_width.data = data.get_bins('wl_width').repeat(spectrum_num)
        elif spectrum_num == wl_width_num:
            self.wl_width.data = data.get_bins('wl_width')
        else:
            self.wl_width.data = torch.tensor([data.select_wavelength(wl).wl_width[0] \
                                               for wl in data.get_bins('wavelength')])

        self.phase_num.data = torch.tensor(len(data.get_bins('phase')))
        self.mod_num.data = torch.tensor(len(data.get_bins('mod')))

    def forward(self, data):
        if self.phase_num.data == 0:
            self.initialize(data)
        return (torch.stack() * self.wl_width).repeat(self.phase_num*self.mod_num)

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

class FittingModel(nn.Module):
    def __init__(self):
        super(FittingModel, self).__init__()
        self.earth = PlanetBlackBody()
        self.instrument = MiYinBasicType()
        self.noise = ShotNoise()

    def forward(self, data):
        return (self.earth(data) + self.noise(data)) * self.instrument(data)


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        self.amp = FittingModel()
        self.__dataset = None
        self.__free_param_list = {}
        self.update_param_list()
        self.__param_list = self.__free_param_list
        
        print("Initial Parameters:")
        for name, param in self.__param_list.items():
            print(f"{name}: {param.item()}")

    def set_data(self, data):
        self.__dataset = data

    def update_param_list(self):
        self.__free_param_list = {}
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                self.__free_param_list[name] = param
    
    def get_initial_values(self):
        return [param.item() for param in self.__free_param_list.values()]
    
    def fix_param(self, param_name):
        if type(param_name) == str:
            param_name = [param_name]
        for name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False
                print(f"fix parameter {name}")
        self.update_param_list()

    def free_param(self, param_name):
        if type(param_name) == str:
            param_name = [param_name]
        for name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = True
                print(f"fix parameter {name}")
        self.update_param_list()

    def set_param_val(self, name, val):
        if type(val) == torch.Tensor:
            self.__param_list[name].data = val
        elif type(val) in [int, float, list, np.ndarray]:
            self.__param_list[name].data = torch.tensor(val)
        else:
            raise TypeError(f"Type {type(val)} not supported!")

    def forward(self):
        predicted_value = self.amp(self.__dataset)
        return torch.sum(predicted_value - self.__dataset.photon_electron * torch.log(predicted_value))

    def objective(self, params):
        if self.__dataset == None:
            raise ValueError('Dataset should be initialized before call the NLL!')
        for val, par in zip(params, self.__free_param_list.values()):
            par.data = torch.tensor(val)
        NLL = self.forward()
        grad = torch.stack(atg.grad(NLL, self.__free_param_list.values()), dim=0)
        return NLL.cpu().detach().numpy(), grad.cpu().detach().numpy()

# 初始化NLL
NLL = NegativeLogLikelihood()
NLL.set_data(data)

# 设置ra, dec扫描范围
fov = np.array([-200., 200.], dtype=np.float64) / cons._radian_to_mas
fov_bins = 20

ra = torch.tensor(np.linspace(fov[0], fov[1], fov_bins))
dec = torch.tensor(np.linspace(fov[0], fov[1], fov_bins))
ra_grid, dec_grid = torch.meshgrid(ra, dec, indexing='ij')

points = torch.stack((ra_grid.flatten(), dec_grid.flatten()), -1)
nll_grid = np.zeros(len(points), dtype=np.float64)

# 固定RA, DEC
NLL.fix_param(['amp.earth.ra', 'amp.earth.dec'])
initial_val = NLL.get_initial_values()
print(f"initial_val: {initial_val}")
bounds = [(6000000, 7000000), (250., 300.), (0., 1e10)]

# 计时
import time
start_time = time.time()
# 扫描ra-dec
result = None
best_point = []
for i, point in tqdm(enumerate(points)):
    #NLL.amp.earth.ra.data = torch.tensor(point[0])
    #NLL.amp.earth.dec.data = torch.tensor(point[1])    
    NLL.set_param_val('amp.earth.ra', point[0])
    NLL.set_param_val('amp.earth.dec', point[1])
    flag = False                                      
    retry_times = 0
    while(not flag):                                  
        init_val = [np.random.uniform(low=val[0], high=val[1]) for val in bounds]
        this_result = minimize(NLL.objective, x0=init_val, bounds=bounds, method='L-BFGS-B', jac=True, 
                               options={'maxcor': 100, 'ftol': 1e-15, 'maxiter': 100000, 'maxls': 50})
        flag = this_result.success
        if flag == False:
            print("Fail to find the minimum result, retry.")
            retry_times += 1
        if(retry_times > 1000):
            print("All retry fails! Move to the next point.")
            break
    #init_val = [np.random.uniform(low=val[0], high=val[1]) for val in bounds]
    #this_result = minimize(NLL.objective, x0=init_val, bounds=bounds, method='L-BFGS-B', jac=True, 
    #                       options={'maxcor': 100, 'ftol': 1e-15, 'maxiter': 100000, 'maxls': 50})
    nll_grid[i] = this_result.fun
    if result == None:
        result = this_result
        best_point = point
    elif this_result.fun < result.fun:
        result = this_result
        best_point = point
        print(f'find new minimum at iter {i}: NLL: {result.fun}')

cov_matrix = result.hess_inv.todense()

end_time = time.time()
print(f"扫描用时：{(end_time-start_time)*1e3} ms")

print(f"HESSE Matrix:\n{cov_matrix}")

# 打印拟合结果
print("扫描结果:")
print(f"radius:\t{NLL.amp.earth.radius.item():6.03f} +/- {np.sqrt(cov_matrix[0,0]):6.03f}")
print(f"temperature:\t{NLL.amp.earth.temperature.item():6.03f} +/- {np.sqrt(cov_matrix[1,1]):6.03f}")
print(f"ra:\t{best_point[0]*cons._radian_to_mas:6.03f}")
print(f"dec:\t{best_point[1]*cons._radian_to_mas:6.03f}")
print(f"scale:\t{NLL.amp.noise.scale.item():6.03e} +/- {np.sqrt(cov_matrix[2,2])*cons._radian_to_mas:6.03e}")
print(f'NLL: {result.fun}')

# basinhopping在扫描结果周边搜索最优值
NLL.free_param(['amp.earth.ra', 'amp.earth.dec'])
initial_val = NLL.get_initial_values()
print(f"initial_val: {initial_val}")
fov_half_binwidth = (fov[1] - fov[0]) / fov_bins / 2.
bounds_ra = (best_point[0]-fov_half_binwidth, best_point[0] + fov_half_binwidth)
bounds_dec = (best_point[1]-fov_half_binwidth, best_point[1] + fov_half_binwidth)
bounds = [(6000000, 7000000), (250., 300.), bounds_ra, bounds_dec, (0., 1e10)]

start_time = time.time()
result = basinhopping(NLL.objective, 
                      x0=initial_val, 
                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds': bounds, 'jac': True, 
                                        'options': {'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50}}, 
                      disp=True)
end_time = time.time()
cov_matrix = result.lowest_optimization_result.hess_inv.todense()
print(f"最小化用时：{(end_time-start_time)*1e3} ms")
# 打印拟合结果
print("拟合结果:")
print(f"radius:\t{NLL.amp.earth.radius.item():6.03f} +/- {np.sqrt(cov_matrix[0,0]):6.03f}")
print(f"temperature:\t{NLL.amp.earth.temperature.item():6.03f} +/- {np.sqrt(cov_matrix[1,1]):6.03f}")
print(f"ra:\t{NLL.amp.earth.ra.item()*cons._radian_to_mas:6.03f} +/- {np.sqrt(cov_matrix[2,2])*cons._radian_to_mas:6.03f}")
print(f"dec:\t{NLL.amp.earth.dec.item()*cons._radian_to_mas:6.03f} +/- {np.sqrt(cov_matrix[3,3])*cons._radian_to_mas:6.03f}")
print(f"scale:\t{NLL.amp.noise.scale.item():6.03e} +/- {np.sqrt(cov_matrix[4,4])*cons._radian_to_mas:6.03e}")
print(f'NLL: {result.fun}')


# Draw NLL distribution
ra_result = NLL.amp.earth.ra.item()*cons._radian_to_mas
dec_result = NLL.amp.earth.ra.item()*cons._radian_to_mas
ra_err = np.sqrt(cov_matrix[2,2])*cons._radian_to_mas
dec_err = np.sqrt(cov_matrix[3,3])*cons._radian_to_mas

ra_grid_numpy = ra_grid.cpu().detach().numpy()
dec_grid_numpy = dec_grid.cpu().detach().numpy()
nll_grid = (nll_grid-np.max(nll_grid)).reshape(fov_bins, fov_bins)
fig, ax = plt.subplots()
levels = np.arange(np.min(nll_grid)-np.min(nll_grid)*0.001, np.max(nll_grid)+np.max(nll_grid)*0.001, np.fabs(np.max(nll_grid)-np.min(nll_grid))/100.)
#levels = np.arange(np.min(nll_grid), np.max(nll_grid), np.fabs(np.max(nll_grid)-np.min(nll_grid))/100.)
trans_map_cont = ax.contourf(ra_grid_numpy*cons._radian_to_mas, dec_grid_numpy*cons._radian_to_mas, nll_grid, levels=levels, cmap = plt.get_cmap("bwr"))
ax.set_xlabel("ra / mas")
ax.set_ylabel("dec / mas")

ax.scatter(ra_result, dec_result, marker='+', color='green')

cbar = fig.colorbar(trans_map_cont)

plt.savefig('fig/maximum_likelihood_shotnoise.pdf')
plt.show()