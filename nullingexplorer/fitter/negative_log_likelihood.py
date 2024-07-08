import torch
import torch.nn as nn
import torch.autograd as atg

import numpy as np
from tensordict import TensorDict

from nullingexplorer.utils import Constants as cons

from abc import ABC, abstractmethod

class NegativeLogLikelihood(nn.Module, ABC):
    def __init__(self, amp: nn.Module, data: TensorDict):
        super().__init__()
        self.amp = amp
        self.dataset = data

        self.__free_param_list = {}
        self.update_param_list()
        self.__param_list = self.__free_param_list

        print("Initial Parameters:")
        for name, param in self.__param_list.items():
            print(f"{name}: {param.item()}")

    def update_param_list(self):
        self.__free_param_list = {}
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                self.__free_param_list[name] = param
    
    def get_param_values(self):
        return [param.item() for param in self.__free_param_list.values()]
    
    def fix_param(self, param_name):
        if type(param_name) == str:
            param_name = [param_name]
        for name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False
                print(f"fix parameter {name}")
        self.update_param_list()

    def free_param(self, param_name):
        if type(param_name) == str:
            param_name = [param_name]
        for name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = True
                print(f"free parameter {name}")
        self.update_param_list()

    def set_param_val(self, name, val):
        if type(val) == torch.Tensor:
            self.__param_list[name].data = val
        elif type(val) in [int, float, list, np.float32, np.float64, np.ndarray]:
            self.__param_list[name].data = torch.tensor(val)
        else:
            raise TypeError(f"Type {type(val)} not supported!")

    def forward(self):
        return self.call_nll()

    @abstractmethod
    def call_nll(self):
        return -1

    def objective(self, params):
        if self.dataset == None:
            raise ValueError('Dataset should be initialized before call the NLL!')
        for val, par in zip(params, self.__free_param_list.values()):
            par.data.fill_(val)
            #par.data = torch.tensor(val)
        NLL = self.call_nll()
        #return NLL.cpu().detach().numpy()
        grad = torch.stack(atg.grad(NLL, self.__free_param_list.values()), dim=0)
        return NLL.cpu().detach().numpy(), grad.cpu().detach().numpy()

    def func_call(self, *args):
        if len(args) != len(self.__free_param_list.keys()):
            raise ValueError("Argument count mismatch. Ensure the number of args matches the number of values in free_params.")

        for key, val in zip(self.__free_param_list.keys(), args):
            path = key.split('.')
            sub_model = self
            for ip in path[:-1]:
                sub_model = getattr(sub_model, ip)
            delattr(sub_model, path[-1])
            setattr(sub_model, path[-1], val)
        #for par, val in zip(self.parameters(), args):
        #for par, val in zip(self.__free_param_list.values(), args):
        #    par.data.fill_(val.data.item())
        return self.call_nll()

    def hessian(self):
        hessian = atg.functional.hessian(self.func_call, tuple(self.free_param_list.values()))
        return torch.stack([torch.stack(hs) for hs in hessian], dim=1)

    def inverse_hessian(self):
        return torch.inverse(self.hessian())

    def std_error(self):
        std_err = torch.sqrt(torch.diag(self.inverse_hessian()))
        return std_err

    @property 
    def free_param_list(self) -> dict:
        return self.__free_param_list

class GaussianNLL(NegativeLogLikelihood):
    def __init__(self, amp: nn.Module, data: TensorDict):
        super().__init__(amp, data)

    def call_nll(self):
        # Gaussian distribution.
        return torch.sum((self.dataset.photon_electron-self.amp(self.dataset))**2/self.dataset.pe_uncertainty**2)

    def call_nll_nosig(self):
        # Only for Chopped design. Assuming that all the backgrounds are shot noises.
        return torch.sum((self.dataset.photon_electron)**2/self.dataset.pe_uncertainty**2)

class PoissonNLL(NegativeLogLikelihood):
    def __init__(self, amp: nn.Module, data: TensorDict):
        super().__init__(amp, data)

    def call_nll(self):
        # Poisson distribution
        predicted_value = self.amp(self.dataset)
        return torch.sum(predicted_value - self.dataset.photon_electron * torch.log(predicted_value))
