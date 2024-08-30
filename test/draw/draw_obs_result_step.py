import sys
sys.path.append('..')

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import mplhep

from iminuit import Minuit, cost
from scipy import stats

plt.style.use(mplhep.style.LHCb2)

nbins = 20

# 30m baseline
#hdf5_path = "results/PoissonObs_20240821_104947/toy_nll_distribution.hdf5"
# 10m baseline
hdf5_path = "results/PoissonObs_20240821_113732/toy_nll_distribution.hdf5"

result = {}
file = h5py.File(hdf5_path, "r")
dataset_list = file.keys()
for ds in dataset_list:
    result[ds] = file[ds][()]
    #print(f"{ds}: \n{result[ds]}")
file.close()

draw_keys = [   
    'sigma', 
    'Rp', 
    'Tp', 
    'true_ang_sep']
draw_range = [
    [0., 50.],
    [0., 5.],
    [0., 1000.],
    [0., 100.]
]
scale = [
    1.,
    1.,
    1.,
    1000.
]
x_title = [
    'Significance',
    'Radius [$R_{{\\rm earth}}$]',
    "Temperature [K]",
    'Angular Separation [mas]'
]
#obs_index = np.where(result['sigma'] >= 10)

def draw_obs(istar):

    total_index = np.where(result['Nstar'] <= istar)
    obs_index = np.where((result['sigma'] >= 10) & (result['Nstar'] <= istar))

    fig = plt.figure(figsize=(12, 10))
    for i, key in enumerate(draw_keys):
        ax = fig.add_subplot(2, 2, i+1)
        counts, bins = np.histogram(result[key][total_index]*scale[i], bins=nbins, range=(draw_range[i][0], draw_range[i][1]))
        obs_counts, obs_bins = np.histogram(result[key][obs_index]* scale[i], bins=nbins, range=(draw_range[i][0], draw_range[i][1]))

        ax.hist(bins[:-1], bins, weights=counts, color='b', label='Total')
        ax.hist(obs_bins[:-1], obs_bins, weights=obs_counts, color='r', label='Observed')

        ax.set_xlabel(x_title[i])
        ax.set_xlim(draw_range[i][0], draw_range[i][1])
        ax.legend(fontsize='x-large')

    #plt.show()
    plt.savefig(f"results/obs_predict/obs_predict_{istar}.jpg")
    plt.close()

if __name__ == '__main__':
    n_star = np.max(result['Nstar']) + 1

    for i in tqdm(range(n_star)):
        draw_obs(i)