import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from scipy.interpolate import griddata

import torch
import torch.nn as nn
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)
from tensordict.prototype import tensorclass

from nullingexplorer.model.amplitude import *
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.io.data import MiYinData


#from tensordict import TensorDict
#import torch
#data = TensorDict({
#    "abc": torch.ones(3, 4, 5),
#    "def": torch.zeros(3, 4, 5, dtype=torch.bool),
#}, batch_size=[3, 4])
#print(data)
#print(data["abc"])

diff_data = MiYinData.load('diff_data.hdf5')
diff_data = diff_data.flatten()

#print(diff_data)
#diff_data.draw(path='fig',draw_err=True, show=True)

dir(MiYinData)


#phase_bins = 360
#spectrum_bins = 30
#pe_np = diff_data.photon_electron.cpu().detach().numpy().reshape(phase_bins, spectrum_bins)
#pe_err_np = diff_data.pe_uncertainty.cpu().detach().numpy().reshape(phase_bins, spectrum_bins)
#phase_np = diff_data.phase.cpu().detach().numpy()
#wavelength_np = diff_data.wavelength.cpu().detach().numpy()
##
## draw pe
#fig, ax = plt.subplots(2,1, figsize=(24.,7.))
#
#print(phase_np)
#print(wavelength_np)
#print(pe_np)
#
#draw_data = ax[0].pcolormesh(phase_np, wavelength_np, pe_np, cmap = plt.get_cmap("bwr"))
#ax[0].set_xlabel("phase / rad")
#ax[0].set_ylabel("wavelength / m")
#cbar = fig.colorbar(draw_data, label="Photon electron", aspect=5, pad=0.01)
#
#draw_err = ax[1].pcolormesh(phase_np, wavelength_np, pe_err_np, cmap = plt.get_cmap("bwr"))
#ax[1].set_xlabel("phase / rad")
#ax[1].set_ylabel("wavelength / m")
#cbar = fig.colorbar(draw_err, label="Uncertainty", aspect=5, pad=0.01)
#
#plt.show()