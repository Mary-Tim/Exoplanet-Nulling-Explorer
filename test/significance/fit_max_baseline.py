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

baseline = np.array([30., 40., 50., 75., 100.], dtype=np.float32)
#baseline = np.array([30., 40., 50., 75., 100., 200.], dtype=np.float32)
versus_baseline = 1. / baseline
max_angular = [54.758256, 41.250716, 33.214278, 22.960869, 17.456898]
max_significance = [10.240890, 9.140695, 8.267168, 6.612447, 5.461815]
#max_angular = [54.758256, 41.250716, 33.214278, 22.960869, 17.456898, 8.926741]
#max_significance = [10.240890, 9.140695, 8.267168, 6.612447, 5.461815, 3.110257]
angular_separation = np.linspace(0., 0.04, 200)

def fit_model(data, param):
    #print(shift)
    return param[1]*np.power(data, param[0]) + param[2]
    #return param[1]*(np.log(200*data) / np.log(param[0]))

## x^a
init_val = [0.7, 140., 1.]
boundary = [(0.1, 0.9), (50., 200.), (0., 10.)]

# log_a(x)
#init_val = [5, 10.]
#boundary = [(3., 5.), (5., 50.)]

def loss(param):
    #return np.fabs(np.sum(max_significance - fit_model(baseline, param)))
    return np.fabs(np.sum(max_significance - fit_model(versus_baseline, param)))
    #return np.fabs(np.sum(max_significance - fit_model(param[0], max_angular)))

#result = minimize(loss, 
#            x0=init_val, 
#            method = 'L-BFGS-B',
#            bounds = boundary,
#            #options={'maxcor': 100, 'ftol': 10, 'maxiter': 10000, 'maxls': 50}
#            )
result = basinhopping(loss, 
            x0=init_val, 
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': boundary, 'jac': False, 
                                'options': {'maxcor': 1000, 'ftol': 1e-10, 'maxiter': 100000, 'maxls': 50}}, 
            stepsize=0.1,
            niter=10000,
            niter_success=500)

fig, ax = plt.subplots()
#ax.scatter(max_angular, max_significance, marker='*', color='black')
#ax.scatter(versus_baseline, max_significance, marker='*', color='black')
ax.scatter(versus_baseline, max_angular, marker='*', color='black')
#ax.scatter(baseline, max_significance, marker='*', color='black')
#ax.plot(angular_separation, fit_model(angular_separation, result.x), color='b')

print(f"{result.x}")

ax.set_xlabel("1 / Baseline [$\\rm{{m^{{-1}}}}$]")
ax.set_ylabel("Angular / mas")
#ax.set_ylabel("Significance")

#ax.legend()

plt.show()




