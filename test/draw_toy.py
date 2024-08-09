import sys
sys.path.append('..')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

import h5py
import numpy as np
from matplotlib import pyplot as plt
import mplhep

from iminuit import Minuit, cost
from scipy import stats

plt.style.use(mplhep.style.LHCb2)

nbins = 20
draw_sigma = 4
init_val = {
    'earth.r_radius':       {'mu': 0., 'sigma': 0.2}, 
    'earth.r_temperature':  {'mu': 0., 'sigma': 0.1}, 
    'earth.r_angular':      {'mu': 0., 'sigma': 0.002}, 
    'earth.r_polar':        {'mu': 0., 'sigma': 0.002}
}
limits = {
    'earth.r_radius':       {'mu': (-0.50, 0.50), 'sigma': (5e-3, 1e-0)}, 
    'earth.r_temperature':  {'mu': (-0.20, 0.20), 'sigma': (5e-3, 1e-0)}, 
    'earth.r_angular':      {'mu': (-0.02, 0.02), 'sigma': (5e-4, 5e-1)}, 
    'earth.r_polar':        {'mu': (-0.02, 0.02), 'sigma': (5e-4, 5e-1)}
}
#hdf5_path = "results/Toy_20240806_173118/toy_MC_result.hdf5"
hdf5_path = "results/Toy_20240807_163626/toy_MC_result.hdf5"
#hdf5_path = "results/Toy_20240808_172237/toy_MC_result.hdf5"

def fit_gauss(data, init_mu=0., init_sigma=0.1):
    def pdf(x, mu, sigma):
        return stats.norm.pdf(x, mu, sigma)
    nll = cost.UnbinnedNLL(data, pdf)
    m = Minuit(nll, mu=init_mu, sigma=init_sigma)
    return m

def to_polar(x, y):
    return np.sqrt((x**2 + y**2)), np.arctan2(y, x)

result = {}
file = h5py.File(hdf5_path, "r")
dataset_list = file.keys()
for ds in dataset_list:
    result[ds] = file[ds][()]
file.close()

if 'param_name' in result.keys():
    param_name = [name.decode("utf-8") for name in result['param_name']]
    result['param_name'] = param_name

truth_convert = np.zeros((len(result["fitted_val"]), 4))
truth_convert[:, 0] = result["true_val"][:, 0] / 6371e3
truth_convert[:, 1] = result["true_val"][:, 1] / 285.
truth_convert[:, 2], truth_convert[:, 3] = to_polar(result["true_val"][:, 2], result["true_val"][:, 3])
truth_convert[:, 2] = truth_convert[:, 2] / 100.

truth_convert[truth_convert[:,3]<0, 3] = truth_convert[truth_convert[:,3]<0, 3] + 2*np.pi

fig = plt.figure(figsize=(12, 10))
num_of_data = len(result["fitted_val"])
for i, name in enumerate(result["param_name"]):


    # fit with gaussian
    residual = result["fitted_val"][:,i]-truth_convert[:,i]
    m = fit_gauss(residual, init_mu=init_val[name]['mu'], init_sigma=init_val[name]['sigma'])
    m.limits['mu'] = limits[name]['mu']
    m.limits['sigma'] = limits[name]['sigma']
    m.migrad()
    m.hesse()

    # Draw distribution
    range_lo = m.values["mu"] - m.values["sigma"] * draw_sigma
    range_hi = m.values["mu"] + m.values["sigma"] * draw_sigma

    ax = fig.add_subplot(2, 2, i+1)
    counts, bins = np.histogram(residual, bins=nbins, range=(range_lo, range_hi))
    errors = np.sqrt(counts)
    ax.errorbar(bins[:-1], counts, yerr=errors, fmt='ok')
    #ax.stairs(counts, bins)
    ax.set_xlabel(name)

    bin_width = (range_hi - range_lo) / float(nbins)
    x_line = np.linspace(range_lo, range_hi, 100)
    ax.plot(x_line, num_of_data*bin_width*stats.norm.pdf(x_line, m.values["mu"], m.values["sigma"]), c='r')
    plt.text(0.6, 0.8, f"$\\mu$: {m.values['mu']:.2e} \n$\\sigma$: {m.values['sigma']:.2e}", fontsize=20, transform=ax.transAxes)

    print(f"{result['param_name'][i]}:  \tmu: {m.values['mu']:.3e},\tsigma: {m.values['sigma']:.3e}")

plt.show()
