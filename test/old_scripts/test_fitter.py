import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd as atg
from scipy.optimize import basinhopping
#torch.set_default_device('cpu')
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.io import DataHandler, FitResult
from nullingexplorer.generator import ObservationCreator
from nullingexplorer.fitter import GaussianNLL
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.utils import get_amplitude


# Observation plan
obs_config = {
        'Spectrum':
            {
            'Type': 'Equal',
            'BinNumber': 30,
            'Low': 7.,
            'High': 18.,        # unit: micrometer
            },
        'Observation':
            {
            'ObsNumber': 360,
            'IntegrationTime': 300,  # unit: second
            'ObsMode': [1, -1],  # For chopped nulling
            'Phase':
                {
                'Start' : 0.,
                'Stop': 360.,   # unit: degree
                },
            'Baseline':
                {
                'Type': 'Constant',
                'Value': 15.,  # unit: meter
                },
            },
        }

# Object config
earth_location=np.array([62.5, 78.1]) / cons._radian_to_mas
earth_temperature = 285.
earth_radius = 6371.e3
cfg.set_property('target_latitude', 30.)
cfg.set_property('target_longitude', 0.)

# Formation config
cfg.set_property('mirror_diameter', 3.5)
cfg.set_property('ratio', 6.)

# Nulling depth config
depth_index = 5
cfg.set_property('nulling_depth', 0.)

# Saving path
from datetime import datetime
import os
start_time = datetime.now()
output_path = f"results/Job_NDz_{start_time.strftime('%Y%m%d_%H%M%S')}"

# 设置ra, dec扫描范围
fov = np.array([-2., 2.], dtype=np.float64)
scan_num = 50

obs_creator = ObservationCreator()
obs_creator.load(obs_config)
data = obs_creator.generate()

# 设置传输图
cfg.set_property('trans_map', 'DualChoppedDestructive')
class Amplitude(BaseAmplitude):
    def __init__(self):
        super(Amplitude, self).__init__()
        self.earth = PlanetBlackBody()
        self.star = StarBlackBodyMatrix()
        self.local_zodi = LocalZodiacalDustMatrix()
        self.exo_zodi = ExoZodiacalDustMatrix()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.earth(data) + self.star(data) + self.local_zodi(data) + \
                self.exo_zodi(data)) * self.instrument(data)

amp = Amplitude()
amp.earth.ra.data = torch.tensor(earth_location[0])
amp.earth.dec.data = torch.tensor(earth_location[1])
amp.earth.temperature.data = torch.tensor(earth_temperature)
amp.earth.radius.data = torch.tensor(earth_radius)

#exit(0)

# 计时
import time
start_time = time.time()
data['photon_electron'] = torch.poisson(amp(data))
data_handler = DataHandler(data)
diff_data = data_handler.diff_data(obs_creator)
#diff_data = data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)[:,:,0]
#data_mod3 = data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)[:,:,0]
#data_mod4 = data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)[:,:,1]
#diff_data['photon_electron'] = (data_mod3['photon_electron'] - data_mod4['photon_electron']).reshape(obs_creator.obs_num,obs_creator.spec_num)
#diff_data['pe_uncertainty'] = torch.sqrt(data_mod3['photon_electron'] + data_mod4['photon_electron']).reshape(obs_creator.obs_num,obs_creator.spec_num)
#diff_data['pe_uncertainty'][diff_data['pe_uncertainty'] == 0] = 1e10
#diff_data.save('results/data_fitter.hdf5')

end_time = time.time()
print(f"数据产生用时：{(end_time-start_time)} s")

cfg.set_property('trans_map', 'DualChoppedDifferential')

class DiffAmplitude(BaseAmplitude):
    def __init__(self):
        super(DiffAmplitude, self).__init__()
        self.earth = get_amplitude('RelativePlanetBlackBody')()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return self.earth(data) * self.instrument(data)

# 初始化NLL
NLL = GaussianNLL(amp=DiffAmplitude(), data=diff_data)

ra_result = np.zeros(scan_num, dtype=np.float64)
dec_result = np.zeros(scan_num, dtype=np.float64)
nll_grid = np.zeros(scan_num, dtype=np.float64)

# 固定RA, DEC
initial_val = NLL.get_param_values()
bounds = [(0.1, 3.), (0.1, 3.), tuple(fov), tuple(fov)]
bounds_lo = np.array([bound[0] for bound in bounds])
bounds_hi = np.array([bound[1] for bound in bounds])

# 计时
start_time = time.time()
# 扫描ra-dec
result = None
for i in tqdm(range(scan_num)):
    flag = False                                      
    retry_times = 0
    while(not flag):                                  
        init_val = np.random.uniform(low=bounds_lo, high=bounds_hi)
        #this_result = minimize(NLL.objective, x0=init_val, bounds=bounds, method='L-BFGS-B', jac=True, 
        #                       options={'maxcor': 20, 'ftol': 1e-15, 'gtol':1e-15, 'maxiter': 1000000, 'maxfun':1000000, 'maxls': 1000000})
        this_result = basinhopping(NLL.objective, 
                      x0=init_val, 
                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds': bounds, 'jac': True, 
                                        'options': {'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50}}, 
                      stepsize=1.,
                      niter=1000,
                      niter_success=50)
        flag = this_result.success
        if flag == False:
            #print(f"Fail to find the minimum result, retry. NLL: {this_result.fun}")
            retry_times += 1
        if(retry_times > 100):
            print("All retry fails! Move to the next point.")
            break
    nll_grid[i] = this_result.fun
    ra_result[i] = this_result.x[2]
    dec_result[i] = this_result.x[3]
    if result == None:
        result = this_result
    elif this_result.fun < result.fun:
        result = this_result
        #print(f'find new minimum at iter {i}: NLL: {result.fun}')

end_time = time.time()
print(f"扫描用时：{(end_time-start_time)} s")

initial_val = result.x
print(f"initial_val: {initial_val}")
start_time = time.time()
result = basinhopping(NLL.objective, 
                      x0=initial_val, 
                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds': bounds, 'jac': True, 
                                        'options': {'maxcor': 20, 'ftol': 1e-15, 'gtol':1e-15, 'maxiter': 1000000, 'maxfun':1000000, 'maxls': 1000000}}, 
                      stepsize=0.1,
                      niter=10000,
                      niter_success=200,
                      disp=False)
end_time = time.time()
print(f"最小化用时：{(end_time-start_time)} s")

# 创建结果目录，存储计算结果
if not os.path.exists(output_path):
    os.mkdir(output_path)
fit_result = FitResult()
fit_result.save_fit_result(nll_model=NLL, scipy_result=result)
# 计算显著度
nll_nosig = NLL.call_nll_nosig().cpu().detach().numpy()
significance = fit_result.significance(bkg=nll_nosig, ndf=8)
# 打印拟合结果
print("拟合结果:")
fit_result.print_result()
print(f"显著度: {significance:.3f}")
fit_result.set_item('ra_grid', ra_result*cons._radian_to_mas)
fit_result.set_item('dec_grid', dec_result*cons._radian_to_mas)
fit_result.set_item('nll_grid', nll_grid)
#fit_result.set_item('significance', significance)
fit_result.save(f'{output_path}/result_fitter_random.hdf5')

nll_grid = nll_grid-nll_nosig

# NLL distribution
fig, ax = plt.subplots()
line = np.arange(0,len(nll_grid))
scat = ax.scatter(line, nll_grid, s=30)
ax.set_xlabel("Task")
ax.set_ylabel("$\\Delta$ LL")
plt.savefig(f'{output_path}/NLL.pdf')

# 排除非正NLL
sel_index = np.where(nll_grid < 0)
ra_result = ra_result[sel_index]
dec_result = dec_result[sel_index]
nll_grid = nll_grid[sel_index]
# NLL从大到小排序
seq_index = np.argsort(-nll_grid)
ra_result = ra_result[seq_index]
dec_result = dec_result[seq_index]
nll_grid = nll_grid[seq_index]
fig, ax = plt.subplots()
levels = np.arange(np.min(nll_grid)*1.005, 10., np.fabs(np.max(nll_grid)-np.min(nll_grid))/100.)
#trans_map_cont = ax.contourf(ra_result*cons._radian_to_mas, dec_result*cons._radian_to_mas, nll_grid, levels=levels, cmap = plt.get_cmap("bwr"))
scat = ax.scatter(ra_result*100., dec_result*100., s=30, c=nll_grid, cmap=plt.get_cmap("bwr"))
fig.colorbar(scat,ax=ax,orientation='vertical',label='$\\Delta$ LL')
ax.set_xlabel("ra / mas")
ax.set_ylabel("dec / mas")
plt.savefig(f'{output_path}/fitter_random.pdf')


#cbar = fig.colorbar(trans_map_cont)

plt.show()