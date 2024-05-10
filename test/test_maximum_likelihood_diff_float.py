import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from scipy.interpolate import griddata
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd as atg
from scipy.optimize import minimize, basinhopping, brute
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)
from tensordict.prototype import tensorclass

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg

# Observation plan
phase_range=[0., np.pi*2]
phase_bins = 90
spectrum_range = np.array([7., 18.], dtype=np.float64) * 1e-6
spectrum_bins = 20
integral_time = 278.*2

# Object config
earth_location=np.array([100./np.sqrt(2), 100./np.sqrt(2)]) / cons._radian_to_mac
earth_temperature = 285.
earth_radius = 6371.e3
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

# Formation config
cfg.set_property('mirror_diameter', 3.5)
cfg.set_property('baseline', 15.)
cfg.set_property('ratio', 3.)

# 设置ra, dec扫描范围
fov = np.array([-200., 200.], dtype=np.float64) / cons._radian_to_mac
fov_bins = 10

def significance(ndf=2, sig=-1000, bkg=-1000):

    delta_2ll = 2 * abs(sig-bkg)
    n_sigma = -stats.norm.ppf(stats.chi2.sf(delta_2ll,df=ndf,loc=0,scale=1)/2)
    return n_sigma

@tensorclass
class MiYinData:
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor
    integral_time: torch.Tensor
    photon_electron: torch.Tensor
    pe_uncertainty: torch.Tensor

    def select_mod(self, mod):
        return self[self.mod==mod]

    def select_data(self, key, val):
        return self[getattr(self, key)==val]

    def nparray(self, key):
        return getattr(self, key).cpu().detach().numpy()

    def get_bins(self, key):
        return torch.unique(getattr(self, key))

    def get_bin_number(self, key):
        return torch.tensor(len(torch.unique(getattr(self, key))))

phase = torch.tensor(np.repeat(np.linspace(phase_range[0], phase_range[1], phase_bins),spectrum_bins*2)).flatten()
wavelength = torch.tensor([np.repeat(np.linspace(spectrum_range[0], spectrum_range[1], spectrum_bins),2)]*phase_bins).flatten()
wl_width = torch.ones(phase_bins*spectrum_bins*2)*((spectrum_range[1]-spectrum_range[0])/spectrum_bins)
intg_time = torch.ones(phase_bins*spectrum_bins*2)*integral_time
mod = torch.tensor([np.array([1,-1])]*(phase_bins*spectrum_bins)).flatten()
photon_electron = torch.zeros(phase_bins*spectrum_bins*2)
pe_uncertainty = torch.zeros(phase_bins*spectrum_bins*2)

data = MiYinData(phase=phase, 
                 wavelength=wavelength, 
                 wl_width=wl_width, 
                 mod=mod, 
                 integral_time=intg_time, 
                 photon_electron=photon_electron, 
                 pe_uncertainty=pe_uncertainty, 
                 batch_size=[phase_bins*spectrum_bins*2])

class Amplitude(BaseAmplitude):
    def __init__(self):
        super(Amplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.star = StarBlackBodyFast()
        self.local_zodi = LocalZodiacalDustFast()
        self.exo_zodi = ExoZodiacalDustFast()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        #return self.earth(data) * self.instrument(data)
        #return (self.earth(data)+ self.mars(data) + self.venus(data)) * self.instrument(data)
        return (self.earth(data) + self.star(data) + self.local_zodi(data) + self.exo_zodi(data)) * self.instrument(data)


amp = Amplitude()
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(earth_radius)

data.photon_electron = torch.poisson(amp(data))
diff_data = data.reshape(phase_bins,spectrum_bins,2)[:,:,0]
diff_data.photon_electron = (data.select_mod(1).photon_electron - data.select_mod(-1).photon_electron).reshape(phase_bins,spectrum_bins)
diff_data.pe_uncertainty = torch.sqrt(data.select_mod(1).photon_electron + data.select_mod(-1).photon_electron).reshape(phase_bins,spectrum_bins)
diff_data.pe_uncertainty[diff_data.pe_uncertainty == 0] = 1e10

class DiffAmplitude(BaseAmplitude):
    def __init__(self):
        super(DiffAmplitude, self).__init__()
        self.earth = PlanetBlackBodyDiff()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        #print(self.earth(data))
        return self.earth(data) * self.instrument(data)

class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        self.amp = DiffAmplitude()
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
        elif type(val) in [int, float, list, np.float32, np.float64, np.ndarray]:
            self.__param_list[name].data = torch.tensor(val)
        else:
            raise TypeError(f"Type {type(val)} not supported!")

    def forward(self, data):
        # Gaussian distribution
        return torch.sum((data.photon_electron-self.amp(data))**2/data.pe_uncertainty**2)
        # Poisson distribution
        #predicted_value = self.amp(self.__dataset)
        #return torch.sum(predicted_value - self.__dataset.photon_electron * torch.log(predicted_value))

    def call_nll(self):
        return self.forward(self.__dataset)
        #return torch.sum((self.__dataset.photon_electron-self.amp(self.__dataset))**2/self.__dataset.pe_uncertainty**2)

    def call_nll_nosig(self):
        return torch.sum((self.__dataset.photon_electron)**2/self.__dataset.pe_uncertainty**2)

    def objective(self, params):
        if self.__dataset == None:
            raise ValueError('Dataset should be initialized before call the NLL!')
        for val, par in zip(params, self.__free_param_list.values()):
            par.data = torch.tensor(val)
        NLL = self.call_nll()
        grad = torch.stack(atg.grad(NLL, self.__free_param_list.values()), dim=0)
        return NLL.cpu().detach().numpy(), grad.cpu().detach().numpy()

    def func_call(self, *args):
        for key, val in zip(self.__free_param_list.keys(), args):
            path = key.split('.')
            sub_model = self
            for ip in path[:-1]:
                sub_model = getattr(sub_model, ip)
            delattr(sub_model, path[-1])
            setattr(sub_model, path[-1], val)
        return self.call_nll()

    def hessian(self):
        hessian = atg.functional.hessian(self.func_call, tuple(self.free_param_list.values()))
        return torch.stack([torch.stack(hs) for hs in hessian], dim=1)

    def inverse_hessian(self):
        return torch.inverse(self.hessian())

    def std_error(self):
        std_err = torch.sqrt(torch.diag(self.inverse_hessian()))
        for i, name in enumerate(self.__free_param_list.keys()):
            if name[-3:] in ['.ra', 'dec']:
            #if 'ra' == name[-2:] or 'dec' == name[-3:]:
                std_err[i] = std_err[i]*cons._radian_to_mac
        return std_err

    @property 
    def free_param_list(self):
        return self.__free_param_list


# 初始化NLL
NLL = NegativeLogLikelihood()
NLL.set_data(diff_data)

#hessian = atg.functional.hessian(NLL.func_call, tuple(NLL.free_param_list.values()))
#print(hessian)


ra = torch.tensor(np.linspace(fov[0], fov[1], fov_bins))
dec = torch.tensor(np.linspace(fov[0], fov[1], fov_bins))
ra_grid, dec_grid = torch.meshgrid(ra, dec, indexing='ij')

points = torch.stack((ra_grid.flatten(), dec_grid.flatten()), -1).cpu().detach().numpy()
ra_result = np.zeros(len(points), dtype=np.float64)
dec_result = np.zeros(len(points), dtype=np.float64)
nll_grid = np.zeros(len(points), dtype=np.float64)

# 固定RA, DEC
#NLL.fix_param(['amp.earth.ra', 'amp.earth.dec'])
initial_val = NLL.get_initial_values()
bounds = [(2000000, 10000000), (200., 800.), tuple(fov), tuple(fov)]
fov_half_binwidth = (fov[1] - fov[0]) / fov_bins / 2.

# 计时
import time
start_time = time.time()
# 扫描ra-dec
result = None
best_point = []
for i, point in tqdm(enumerate(points)):
    #NLL.amp.earth.ra.data = torch.tensor(point[0])
    #NLL.amp.earth.dec.data = torch.tensor(point[1])    
    #NLL.set_param_val('amp.earth.ra', point[0])
    #NLL.set_param_val('amp.earth.dec', point[1])
    bounds[2] = (point[0]-fov_half_binwidth, point[0]+fov_half_binwidth)
    bounds[3] = (point[1]-fov_half_binwidth, point[1]+fov_half_binwidth)
    flag = False                                      
    retry_times = 0
    while(not flag):                                  
        init_val = [np.random.uniform(low=val[0], high=val[1]) for val in bounds]
        #this_result = minimize(NLL.objective, x0=init_val, bounds=bounds, method='L-BFGS-B', jac=True, 
        #                       options={'maxcor': 100, 'ftol': 1e-15, 'maxiter': 100000, 'maxls': 50})
        this_result = basinhopping(NLL.objective, 
                      x0=initial_val, 
                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds': bounds, 'jac': True, 
                                        'options': {'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50}}, 
                      stepsize=100.,
                      niter=50,
                      niter_success=20)
        flag = this_result.success
        if flag == False:
            print(f"Fail to find the minimum result, retry. NLL: {this_result.fun}")
            retry_times += 1
        if(retry_times > 1000):
            print("All retry fails! Move to the next point.")
            break
    #init_val = [np.random.uniform(low=val[0], high=val[1]) for val in bounds]
    #this_result = minimize(NLL.objective, x0=init_val, bounds=bounds, method='L-BFGS-B', jac=True, 
    #                       options={'maxcor': 100, 'ftol': 1e-15, 'maxiter': 100000, 'maxls': 50})
    nll_grid[i] = this_result.fun
    ra_result[i] = this_result.x[2]
    dec_result[i] = this_result.x[3]
    if result == None:
        result = this_result
        best_point = point
    elif this_result.fun < result.fun:
        result = this_result
        best_point = point
        print(f'find new minimum at iter {i}: NLL: {result.fun}')

#cov_matrix = result.hess_inv.todense()
#cov_matrix = result.lowest_optimization_result.hess_inv.todense()

end_time = time.time()
print(f"扫描用时：{(end_time-start_time)*1e3} ms")

#print(f"HESSE Matrix:\n{cov_matrix}")

# 打印拟合结果
print("扫描结果:")
print(f"radius:\t{result.x[0]:6.03f}")
print(f"temperature:\t{result.x[1]:6.03f}")
print(f"ra:\t{best_point[0]*cons._radian_to_mac:6.03f}")
print(f"dec:\t{best_point[1]*cons._radian_to_mac:6.03f}")
print(f'NLL: {result.fun}')

# basinhopping在扫描结果周边搜索最优值
NLL.free_param(['amp.earth.ra', 'amp.earth.dec'])
initial_val = result.x
print(f"initial_val: {initial_val}")
fov_half_binwidth = (fov[1] - fov[0]) / fov_bins / 2.
bounds_ra = (best_point[0]-fov_half_binwidth*2, best_point[0] + fov_half_binwidth*2)
bounds_dec = (best_point[1]-fov_half_binwidth*2, best_point[1] + fov_half_binwidth*2)
bounds = [(2000000, 10000000), (200., 800.), bounds_ra, bounds_dec]

start_time = time.time()
result = basinhopping(NLL.objective, 
                      x0=initial_val, 
                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds': bounds, 'jac': True, 
                                        'options': {'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50}}, 
                      stepsize=2.,
                      niter=10000,
                      niter_success=500,
                      disp=True)
end_time = time.time()
#cov_matrix = result.lowest_optimization_result.hess_inv.todense()
#print(cov_matrix)
print(f"最小化用时：{(end_time-start_time)*1e3} ms")

# 利用torch计算hessian矩阵
for name, val in zip(NLL.free_param_list.keys(), result.x):
    NLL.set_param_val(name, val)
std_err = NLL.std_error()
print(std_err)

# 打印拟合结果
print("拟合结果:")
print(f"radius:\t{result.x[0]:6.03e} +/- {std_err[0]:6.03e}")
print(f"temperature:\t{result.x[1]:6.03f} +/- {std_err[1]:6.03f}")
print(f"ra:\t{result.x[2]*cons._radian_to_mac:6.03f} +/- {std_err[2]:6.03f}")
print(f"dec:\t{result.x[3]*cons._radian_to_mac:6.03f} +/- {std_err[3]:6.03f}")
print(f'NLL: {result.fun:.03f}')
nll_nosig = NLL.call_nll_nosig().cpu().detach().numpy()
print(f'NLL without planet: {NLL.call_nll_nosig():.03f}')
print(f'Significance: {significance(ndf=8, sig=result.fun, bkg=nll_nosig):.03f}')

# Save the result
#import h5py
#with h5py.File("results/result_diff.hdf5", 'w') as file:
#    file.create_dataset('result', result.x, dtype='f8')
#    file.create_dataset('std_err', std_err.cpu().detach().numpy(), dtype='f8')
#    file.create_dataset('hees_inv', NLL.inverse_hessian().cpu().detach().numpy())
#    file.create_dataset('ra_grid', ra_grid.cpu().detach().numpy()*cons._radian_to_mac, dtype='f8')
#    file.create_dataset('dec_grid', dec_grid.cpu().detach().numpy()*cons._radian_to_mac, dtype='f8')
#    file.create_dataset('nll_grid', nll_grid)
#    file.close()

# Draw NLL distribution
#ra_result = result.x[2]*cons._radian_to_mac
#dec_result = result.x[3]*cons._radian_to_mac
#ra_err = std_err[2]
#dec_err = std_err[3]

ra_grid_numpy = ra_result.reshape(fov_bins, fov_bins)
dec_grid_numpy = dec_result.reshape(fov_bins, fov_bins)
nll_grid = (nll_grid-np.max(nll_grid)).reshape(fov_bins, fov_bins)
fig, ax = plt.subplots()
levels = np.arange(np.min(nll_grid)*1.005, 10., np.fabs(np.max(nll_grid)-np.min(nll_grid))/100.)
#levels = np.arange(np.min(nll_grid), np.max(nll_grid), np.fabs(np.max(nll_grid)-np.min(nll_grid))/100.)
trans_map_cont = ax.contourf(ra_grid_numpy*cons._radian_to_mac, dec_grid_numpy*cons._radian_to_mac, nll_grid, levels=levels, cmap = plt.get_cmap("bwr"))
ax.set_xlabel("ra / mas")
ax.set_ylabel("dec / mas")

#ax.scatter(ra_result, dec_result, marker='+', color='green')

cbar = fig.colorbar(trans_map_cont)

print(NLL.inverse_hessian().cpu().detach().numpy())
# Save the result
import h5py
with h5py.File("results/result_diff.hdf5", 'w') as file:
    file.create_dataset('result', data=result.x, dtype='f8')
    file.create_dataset('std_err', data=std_err.cpu().detach().numpy(), dtype='f8')
    file.create_dataset('hees_inv', data=NLL.inverse_hessian().cpu().detach().numpy(), dtype='f8')
    file.create_dataset('ra_grid', data=ra_grid.cpu().detach().numpy()*cons._radian_to_mac, dtype='f8')
    file.create_dataset('dec_grid', data=dec_grid.cpu().detach().numpy()*cons._radian_to_mac, dtype='f8')
    file.create_dataset('nll_grid', data=nll_grid, dtype='f8')
    file.close()

plt.savefig('fig/maximum_likelihood_diff_float.pdf')
plt.show()