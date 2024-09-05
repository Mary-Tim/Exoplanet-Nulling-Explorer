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

#hdf5_path = "../results/ObsTime_20240904_144241/toy_nll_distribution.hdf5"
#hdf5_path = "../results/ObsTime_20240904_153753/toy_nll_distribution.hdf5"
# 自由基线
#hdf5_path = "../results/ObsTime_20240905_094248/toy_nll_distribution.hdf5"
#hdf5_path = "../results/ObsTime_20240905_092029/toy_nll_distribution.hdf5"
#hdf5_path = "../results/ObsTime_20240904_174953/toy_nll_distribution.hdf5"
#hdf5_path = "../results/ObsTime_20240904_160655/toy_nll_distribution.hdf5"
# 自由基线, exo_zodi=3
hdf5_path = "../results/ObsTime_20240905_094248/toy_nll_distribution.hdf5"
# 最小基线30m
#hdf5_path = "../results/ObsTime_20240904_165119/toy_nll_distribution.hdf5"

result = {}
file = h5py.File(hdf5_path, "r")
dataset_list = file.keys()
for ds in dataset_list:
    result[ds] = file[ds][()]
    #print(f"{ds}: \n{result[ds]}")
file.close()

draw_keys = [   
    'Baseline', 
    'ObsTime']
    #'AngSep']
draw_range = [
    [10., 45.],
    [0., 100.],
]
scale = [
    1.,
    0.1,
]
x_title = [
    'Baseline [m]',
    'ObsTime [hour]',
]
#obs_index = np.where(np.logical_and(result['sigma'] >= 10, result['Tp'] <= 500.))
#obs_index = np.where(result['sigma'] >= 10 and result['Tp'] <= 300.)
obs_index = np.where(result['ObsTime'] >= 10000.)

long_time_candidates = np.zeros((len(obs_index[0]),5))
print(long_time_candidates)

for i, index in enumerate(obs_index[0]):
    #print(f"Rs: {result['Rs'][index]:.2f},\tTs: {result['Ts'][index]:.2f},\tDs: {result['Ds'][index]:.2f},\tZodi: {result['z'][index]:.2f},\tObsTime: {result['ObsTime'][index]:.2f}")
    long_time_candidates[i,:] = [result['ObsTime'][index], result['Rs'][index], result['Ts'][index], result['Ds'][index], result['z'][index]]

long_time_candidates = long_time_candidates[long_time_candidates[:,0].argsort()]

for item in long_time_candidates:
    print(f"ObsTime: {item[0]:.2f},\tRs: {item[1]:.2f},\tTs: {item[2]:.2f},\tDs: {item[3]:.2f},\tZodi: {item[4]:.2f},\t")

fig = plt.figure(figsize=(12, 5))
for i, key in enumerate(draw_keys):
    ax = fig.add_subplot(1, 2, i+1)
    counts, bins = np.histogram(result[key]*scale[i], bins=nbins, range=(draw_range[i][0], draw_range[i][1]))

    ax.hist(bins[:-1], bins, weights=counts, color='b', label='Total')
    #ax.bar(bins[:-1], counts, color='r', label='Total')
    #ax.bar(obs_bins[:-1], obs_counts, color='b', label='Observed')

    ax.set_xlabel(x_title[i])
    ax.set_xlim(draw_range[i][0], draw_range[i][1])
    #ax.legend(fontsize='x-large')

print(f"Total integral time: {np.sum(result['ObsTime']/10./24.):.3f} [day]")
plt.show()