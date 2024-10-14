import sys
sys.path.append('../..')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

import numpy as np
import h5py

from matplotlib import pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)

from nullingexplorer.model.spectrum import *
from nullingexplorer.generator import ObservationCreator
from nullingexplorer.io.fit_result import FitResult

obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 5.,
        'High': 18.,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 1,
        'IntegrationTime': 200,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 30.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 4,   # Diameter of MiYin primary mirror [meter]
        'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
        'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
        'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
    }
}

interp_num = 12
#hdf5_path = '../results/Job_20240913_093058/result_earth.hdf5'
#hdf5_path = '../results/Job_20240919_151437/result_earth.hdf5'
#hdf5_path = '../results/Job_20240920_110708/result_earth.hdf5'
#hdf5_path = '../results/Job_20240923_155449/result_earth.hdf5'
#hdf5_path = '../results/Job_20240923_163300/result_earth.hdf5'
hdf5_path = '../results/Job_20240923_165130/result_earth.hdf5'

# Linear
#hdf5_path = '../results/Job_20240920_104025/result_earth.hdf5'

result = FitResult.load(hdf5_path)

obs_creator = ObservationCreator()
black_body = RelativeBlackBodySpectrum()
#cspline = LinearInterpolation(wl_min=5, wl_max=17, num_points=interp_num)
cspline = CubicSplineIntegral(wl_min=5.5, wl_max=16.5, num_points=interp_num)
#cspline = CubicSplineInterpolation(wl_min=5, wl_max=17, num_points=interp_num)

tempurature = torch.tensor(273.)

flux_point = np.zeros(interp_num)
flux_err = np.zeros(interp_num)

param_name = result.get_item('param_name')
param_val = result.get_item('param_val')
param_err = result.get_item('std_err')

for i, name in enumerate(param_name):
    tail = name.split('.')[-1]
    if tail.startswith('flux_'):
        iflux = int(tail.split('_')[-1])
        flux_point[iflux] = param_val[i]
        flux_err[iflux] = param_err[i]

cspline.flux.data = torch.tensor(flux_point)

obs_creator.load(obs_config)
data = obs_creator.generate()
obs_config['Spectrum']['R'] = 1000
interp_data = obs_creator.generate()

print(cspline.interp_points.cpu().detach().numpy())
print(flux_point)

fig, ax = plt.subplots()
ax.plot(data['wl_mid'].cpu().detach().numpy(), black_body(data).cpu().detach().numpy(), color='black', label='data')
ax.plot(interp_data['wl_mid'].cpu().detach().numpy(), cspline(interp_data).cpu().detach().numpy(), color='green', label='cubic-spline')
ax.errorbar(cspline.interp_points.cpu().detach().numpy(), flux_point, yerr=flux_err, fmt='ok', markersize=10, capsize=5, label='interpolation points')

print(len(data))

plt.legend()
plt.show()

