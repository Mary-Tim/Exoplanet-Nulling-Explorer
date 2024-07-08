import torch
import torch.nn as nn
import torch.autograd as atg

import numpy as np
from tensordict import TensorDict

from nullingexplorer.utils import Constants as cons
from nullingexplorer.fitter import NegativeLogLikelihood

from abc import ABC, abstractmethod

class ENEFitter():
    def __init__(self, nll_model: NegativeLogLikelihood):
        self.__nll_model = nll_model