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
#obs_config = {
#        'Spectrum':
#            {
#            'Type': 'Equal',
#            'BinNumber': 30,
#            'Low': 7.,
#            'High': 18.,        # unit: micrometer
#            },
#        'Observation':
#            {
#            'ObsNumber': 1,
#            'IntegrationTime': 300,  # unit: second
#            'ObsMode': [1, -1],  # For chopped nulling
#            'Phase':
#                {
#                'Start' : 0.,
#                'Stop': 360.,   # unit: degree
#                },
#            'Baseline':
#                {
#                'Type': 'Constant',
#                'Value': 15.,  # unit: meter
#                },
#            },
#        }
obs_config = {
        'Spectrum':
            {
            'Type': 'Resolution',
            'R': 20,
            'Low': 4.,
            'High': 18.5,        # unit: micrometer
            },
        'Observation':
            {
            'ObsNumber': 1,
            'IntegrationTime': 1.,  # unit: second
            'ObsMode': [1],  # [1] or [-1] or [1, -1]
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
cfg.set_property('target_latitude', 135.)
cfg.set_property('target_longitude', 45.)

# Formation config
cfg.set_property('mirror_diameter', 2.)
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
class AmplitudeMatrix(BaseAmplitude):
    def __init__(self):
        super(AmplitudeMatrix, self).__init__()
        self.exo_zodi = ExoZodiacalDustMatrix()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.exo_zodi(data)) * self.instrument(data)

amp_matrix = AmplitudeMatrix()
result_matrix = amp_matrix(data).cpu().detach().numpy()

class AmplitudeInterp(BaseAmplitude):
    def __init__(self):
        super(AmplitudeInterp, self).__init__()
        self.exo_zodi = ExoZodiacalDust()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.exo_zodi(data)) * self.instrument(data)

amp_interp = AmplitudeInterp()
result_interp = amp_interp(data).cpu().detach().numpy()

plt.plot((data['wl_mid']).cpu().detach().numpy(), result_matrix, 'r-', label='matrix')
plt.plot((data['wl_mid']).cpu().detach().numpy(), result_interp, 'b--', label='interp')

plt.show()
