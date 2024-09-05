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

fit_point = np.array([
    [1 , 1801.89145706],
    [2 , 1784.71436851],
    [3 , 1791.00535393],
    [4 , 1776.30737853],
    [5 , 1761.28848498],
    [6 , 1723.56412456],
    [7 , 1715.25944858],
    [7 , 1715.25944858],
    [8 , 1694.29704737],
    [9 , 1679.83548953],
    [10, 1668.62830787],
    [12, 1631.49730415],
    [14, 1596.73670862],
    [16, 1562.52396168],
    [18, 1540.45738978],
    [20, 1506.59485116]]
    )

interp_array = np.linspace(fit_point[0,0], fit_point[-1,0], 100)

def fit_model(data, param):
    return param[0]*data + param[1]

## x^a
init_val = [-14., 1800.]
boundary = [(-16., -12.), (1790., 1820.)]

def loss(param):
    #return np.fabs(np.sum(max_significance - fit_model(baseline, param)))
    return np.fabs(np.sum(fit_point[:,1] - fit_model(fit_point[:,0], param)))
    #return np.fabs(np.sum(max_significance - fit_model(param[0], max_angular)))

result = basinhopping(loss, 
            x0=init_val, 
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': boundary, 'jac': False, 
                                'options': {'maxcor': 1000, 'ftol': 1e-10, 'maxiter': 100000, 'maxls': 50}}, 
            stepsize=10,
            niter=10000,
            niter_success=500)

fig, ax = plt.subplots()
#ax.scatter(max_angular, max_significance, marker='*', color='black')
#ax.scatter(versus_baseline, max_significance, marker='*', color='black')
ax.scatter(fit_point[:,0], fit_point[:,1], marker='*', color='black')
#ax.scatter(baseline, max_significance, marker='*', color='black')
ax.plot(interp_array, fit_model(interp_array, result.x), color='b')

print(f"{result.x}")

ax.set_xlabel("Distance / pc")
ax.set_ylabel("k")
#ax.set_ylabel("Significance")

correlation_matrix = np.corrcoef(fit_point[:,0], fit_point[:,1])
print(correlation_matrix)
#ax.legend()

plt.show()




