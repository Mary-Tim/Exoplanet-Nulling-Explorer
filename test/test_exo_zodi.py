import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
import time
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

start_time = time.time()
result_matrix = amp_matrix(data).cpu().detach().numpy()
end_time = time.time()
matrix_time = end_time - start_time

class AmplitudeInterp(BaseAmplitude):
    def __init__(self):
        super(AmplitudeInterp, self).__init__()
        self.exo_zodi = ExoZodiacalDust()
        self.instrument = MiYinBasicType()

    def forward(self, data):
        return (self.exo_zodi(data)) * self.instrument(data)

amp_interp = AmplitudeInterp()

start_time = time.time()
result_interp = amp_interp(data).cpu().detach().numpy()
end_time = time.time()
interp_time = end_time - start_time
print(f"Matrix：{matrix_time} s")
print(f"Interp：{interp_time} s")

plt.plot((data['wl_mid']).cpu().detach().numpy(), result_matrix, 'r-', label='matrix')
plt.plot((data['wl_mid']).cpu().detach().numpy(), result_interp, 'b--', label='interp')

plt.show()
