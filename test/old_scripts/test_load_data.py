import sys
sys.path.append('..')

import torch
import torch.nn as nn

from tensordict import TensorDict
from nullingexplorer.io.data import DataHandler

data_handler = DataHandler.load("results/test.hdf5")

print(data_handler.data)