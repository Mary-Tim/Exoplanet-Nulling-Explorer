import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm
from scipy.interpolate import interpn
from scipy.optimize import minimize, basinhopping

from itertools import cycle
cycol = cycle('bgrcmk')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.significance import PoissonSignificance

distance = 20
#1pc, 1801.89145706
#2pc, 1784.71436851
#3pc, 1791.00535393
#4pc, 1776.30737853
#5pc, 1761.28848498
#6pc, 1723.56412456
#7pc, 1715.25944858
#7pc, 1715.25944858
#8pc, 1694.29704737
#9pc, 1679.83548953
#10pc, 1668.62830787
#12pc, 1631.49730415
#14pc, 1596.73670862
#16pc, 1562.52396168
#18pc, 1540.45738978
#20pc, 1506.59485116


# Observation plan
obs_config = {
    'Spectrum':{
        'Type': 'Resolution',
        'R': 20,
        'Low': 5.,
        'High': 17,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 360,
        'IntegrationTime': 200,  # unit: second
        'ObsMode': [1, -1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 0.,
            'Stop': 360.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 30.,  # unit: meter
            #'Type': 'Linear',
            #'Low': 10.,  # unit: meter
            #'High': 50.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'ratio':    6.,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 3.5,   # Diameter of MiYin primary mirror [meter]
        'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
        'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
        'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
    }
}

sig_amp_config = {
    'Amplitude':{
        'earth':{
            'Model': 'PlanetBlackBody',
            'Spectrum': 'InterpBlackBody',
            'Parameters':
            {
                'radius':         {'mean': 6371.e3},
                'temperature':    {'mean': 285.},
                'ra':            {'mean': 100.},
                'dec':            {'mean': 0.},
            },
        },
    },
    'Instrument': {'Model': 'MiYinBasicType'},
    'TransmissionMap': {'Model': 'DualChoppedDestructive'},
    'Configuration':{
        'distance': distance,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

bkg_amp_config = {
    'Amplitude':{
        'star':{
            'Model': 'StarBlackBodyConstant',
        },
        'local_zodi':{
            'Model': 'LocalZodiacalDustConstant',
        },
        'exo_zodi':{
            "Model": 'ExoZodiacalDustConstant',
        },
    },
    'Instrument': {'Model': 'MiYinBasicType'},
    'TransmissionMap': {'Model': 'DualChoppedDestructive'},
    'Configuration':{
        'distance': distance,         # distance between Miyin and target [pc]
        'star_radius': 695500,  # Star radius [kilometer]
        'star_temperature': 5772,   # Star temperature [Kelvin]
        'target_longitude': 0.,     # Ecliptic longitude [degree]
        'target_latitude': 0.,      # Ecliptic latitude  [degree]
        'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
    }
}

angular_separation = np.linspace(1., 100., 200)

sig_poisson = PoissonSignificance()
#sig_poisson.obs_config = obs_config

sig_poisson.bkg_amp_config = bkg_amp_config
#bkg_pe = sig_poisson.gen_bkg_pe()

significance = np.zeros(len(angular_separation))
theta = 30. / 180. * np.pi
baseline = np.array([30., 40., 50., 75., 100.], dtype=np.float32)
#theta_array = np.array([0., 30., 45., 60., 90.], dtype=np.float32) / 180. * np.pi
max_significance = np.zeros(len(baseline))
max_angular = np.zeros(len(baseline))

fig, ax = plt.subplots()
for ib, bl in tqdm(enumerate(baseline), total=len(baseline)):
    obs_config['Observation']['Baseline']['Value'] = bl
    sig_poisson.obs_config = obs_config
    bkg_pe = sig_poisson.gen_bkg_pe()

    init_angular = 0.8 * (1e-5 / bl) * (180. / np.pi) * 3600. *1e3
    boundary = [(0.8 * init_angular, 1.2*init_angular)]

    def eval_significance(angular):
        sig_amp_config['Amplitude']['earth']['Parameters']['ra']['mean'] = angular * np.cos(theta)
        sig_amp_config['Amplitude']['earth']['Parameters']['dec']['mean'] = angular * np.sin(theta)
        sig_pe = sig_poisson.gen_sig_pe(sig_amp_config)
        return float(sig_poisson.get_significance(sig_pe, bkg_pe))

    def negative_significance(x):
        return -eval_significance(x[0])

    result = minimize(negative_significance, 
                x0=init_angular, 
                method = 'L-BFGS-B',
                bounds = boundary,
                options={'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50})

    max_angular[ib] = result.x[0]
    max_significance[ib] = -result.fun
    print(f"Find maximum significance: {-result.fun:2f} at {result.x[0]:2f} mas, with baseline {bl:.0f} m, boundary: {boundary}")
    color = next(cycol)
    ax.plot(angular_separation, np.array([eval_significance(angular) for angular in angular_separation]), color=color, label=f"{bl:.0f}m")
    ax.scatter(result.x[0], -result.fun, marker='*', color='black')

ax.set_xlabel("Angular / mas")
ax.set_ylabel("Significance")

ax.legend()

def fit_model(data, param):
    return param[0]*data
init_val = [1700.]
boundary = [(1400., 1900.)]

def loss(param):
    #return np.fabs(np.sum(max_significance - fit_model(baseline, param)))
    return np.fabs(np.sum(max_angular - fit_model(1./baseline, param)))
result = basinhopping(loss, 
            x0=init_val, 
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': boundary, 'jac': False, 
                                'options': {'maxcor': 1000, 'ftol': 1e-10, 'maxiter': 100000, 'maxls': 50}}, 
            stepsize=100,
            niter=10000,
            niter_success=500)
fig, ax2 = plt.subplots()
#ax.scatter(max_angular, max_significance, marker='*', color='black')
#ax.scatter(versus_baseline, max_significance, marker='*', color='black')
ax2.scatter(1./baseline, max_angular, marker='*', color='black')
#ax.scatter(baseline, max_significance, marker='*', color='black')
bl_array = np.linspace(0.,1./baseline, 100)
ax2.plot(bl_array, fit_model(bl_array, result.x), color='b')

print(f"{result.x}")

ax2.set_xlabel("1 / Baseline [$\\rm{{m^{{-1}}}}$]")
ax2.set_ylabel("Angular / mas")
#ax.set_ylabel("Significance")

correlation_matrix = np.corrcoef(1./baseline, max_angular)
print(correlation_matrix)

plt.show()





