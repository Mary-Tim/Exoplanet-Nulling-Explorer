import sys
sys.path.append('..')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

import numpy as np

from matplotlib import pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)

from nullingexplorer.model.spectrum import *
from nullingexplorer.generator import ObservationCreator

obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 3.,
        'High': 19.,        # unit: micrometer
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

interp_num = 11

obs_creator = ObservationCreator()
black_body = RelativeBlackBodySpectrum()
linear = LinearInterpolation(num_points=interp_num)
#cspline = CubicSplineInterpolation(num_points=interp_num)
cspline = CubicSplineInterpolation(interp_points=np.array([5, 8, 10, 11, 12, 14, 15, 17])*1e-6)

obs_creator.load(obs_config)
data = obs_creator.generate()

flux = black_body(data)

linear.load_spectrum(data['wl_mid'].cpu().detach().numpy(), flux.cpu().detach().numpy())
cspline.load_spectrum(data['wl_mid'].cpu().detach().numpy(), flux.cpu().detach().numpy())
print(cspline.interp_points.cpu().detach().numpy())
print(cspline.flux.data)

fig, ax = plt.subplots()
ax.plot(data['wl_mid'].cpu().detach().numpy(), flux.cpu().detach().numpy(), color='black', label='data')
ax.plot(cspline.interp_points.cpu().detach().numpy(), cspline.flux.cpu().detach().numpy(), color='blue', label='load')
ax.plot(data['wl_mid'].cpu().detach().numpy(), linear(data).cpu().detach().numpy(), color='red', label='linear')
ax.plot(data['wl_mid'].cpu().detach().numpy(), cspline(data).cpu().detach().numpy(), color='green', label='cubic-spline')

print(f"ratio: {cspline(data).cpu().detach().numpy() / flux.cpu().detach().numpy()}")

plt.legend()
plt.show()

