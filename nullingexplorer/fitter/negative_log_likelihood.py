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
        self.__const_param_list = {}
        self.create_boundary()
        self.update_param_list()

        #print(self.__total_gaussian_constraints)

        #print("Initial Parameters:")
        #for name, param in self.__param_list.items():
        #    print(f"{name}: {param.item()}")

    def update_param_list(self):
        self.__free_param_list = {}
        self.__const_param_list = {}
        self.__boundary = {}
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                self.__free_param_list[name] = param
            else:
                self.__const_param_list[name] = param

        self.update_boundary()
        self.update_gaussian_constraints()
    
    def create_boundary(self):
        self.__total_boundary = {}
        self.__total_gaussian_constraints = {}
        for key, val in self.named_parameters():
            sub_model = self
            path = key.split('.')
            for ip in path[:-1]:
                sub_model = getattr(sub_model, ip)
                if hasattr(sub_model, 'boundary'):
                    if path[-1] in sub_model.boundary.keys():
                        self.__total_boundary[key] = sub_model.boundary[path[-1]]
                if hasattr(sub_model, 'gaussian_constraints'):
                    if path[-1] in sub_model.gaussian_constraints.keys():
                        self.__total_gaussian_constraints[key] = sub_model.gaussian_constraints[path[-1]]

    def get_boundary(self, key):
        return self.__total_boundary[key]
    
    def update_boundary(self):
        self.__boundary = {}
        for key in self.__free_param_list.keys():
            self.__boundary[key] = self.__total_boundary[key]

    def update_gaussian_constraints(self):
        if not hasattr(self, '_gaussian_constraints'):
            self._gaussian_constraints = {}
        for key in self.__total_gaussian_constraints.keys():
            if key in self.__free_param_list.keys():
                self._gaussian_constraints[key] = self.__total_gaussian_constraints[key]

        if self._gaussian_constraints == {}:
            del self._gaussian_constraints

    def get_boundaries(self):
        boundaries = []
        for param_name, boundary in self.__boundary.items():
            param = self.__free_param_list[param_name]
            if param.numel() == 1:  # Scalar parameters
                boundaries.append(tuple(boundary.cpu().detach().numpy()))
            else:  # 1D vector parameters
                min_vals, max_vals = boundary
                boundaries.extend(list(zip(min_vals.cpu().detach().numpy(), max_vals.cpu().detach().numpy())))
        return boundaries
    
    def get_param_values(self):
        return [param.item() for param in self.__free_param_list.values()]

    @property
    def name_of_params(self):
        return self.__free_param_list.keys()

    @property
    def name_of_params_flat(self):
        #return self.__free_param_list.keys()
        name_list = []
        for name, param in self.__free_param_list.items():
            if param.numel() == 1:
                name_list.append(name)
            else:
                for i in range(param.numel()):
                    name_list.append(f"{name}_{i}")
        return name_list

    @property
    def num_of_params(self):
        lens = 0
        for param in self.__free_param_list.values():
            lens += param.numel()
        print(f"Num of params: {lens}")
        return lens

    def config_fit_params(self, params:list):
        for key in params:
            if key not in self.__free_param_list.keys():
                print(f"Warning! Cannot find parameter {key} in fit model, skip it.")
                params.remove(key)
        for key in self.__free_param_list.keys():
            if key in params:
                self.free_param(key)
            else:
                self.fix_param(key)
    
    def fix_param(self, param_name):
        if type(param_name) == str:
            param_name = [param_name]
        for name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False
                #print(f"fix parameter {name}")
        self.update_param_list()

    def free_all_params(self):
        for _, param in self.named_parameters():
            param.requires_grad = True
        self.update_param_list()

    def free_param(self, param_name):
        if type(param_name) == str:
            param_name = [param_name]
        for name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = True
                #print(f"free parameter {name}")
        self.update_param_list()

    def set_param_val(self, name, val, constant=False):
        if constant == False:
            if type(val) == torch.Tensor:
                self.__free_param_list[name].data = val
            elif type(val) in [int, float, list, np.float32, np.float64, np.ndarray]:
                self.__free_param_list[name].data.fill_(val)
            else:
                raise TypeError(f"Type {type(val)} not supported!")

        else:
            if type(val) == torch.Tensor:
                self.__const_param_list[name].data = val
            elif type(val) in [int, float, list, np.float32, np.float64, np.ndarray]:
                self.__const_param_list[name].data.fill_(val)
            else:
                raise TypeError(f"Type {type(val)} not supported!")

    def set_gaussian_constraint(self, param_name, mu, sigma):
        if not hasattr(self, '__total_gaussian_constraints'):
            self.__total_gaussian_constraints = {}
        if param_name not in self.name_of_params:
            raise ValueError(f"Parameter {param_name} is not in fit model!")
        self.__total_gaussian_constraints[param_name] = (mu, sigma)
        self.update_gaussian_constraints()

    def forward(self):
        return self.call_nll()

    @abstractmethod
    def call_nll(self):
        # If amp values contain NaN or inf, raise a ValueError exception
        self.amp_val = self.amp(self.dataset)
        if torch.isnan(self.amp_val).any() or torch.isinf(self.amp_val).any():
            raise ValueError("Amp_val contains nan or inf values. Cannot computes the gradient of current amplitude.")

        nll = torch.tensor(0., requires_grad=True)
        gaussian_terms = []
        if hasattr(self, '_gaussian_constraints'):
            for param_name, (mu, sigma) in self._gaussian_constraints.items():
                gaussian_terms.append(0.5 * ((self.__free_param_list[param_name] - mu) / sigma) ** 2)
        if gaussian_terms != []:
            nll = nll + torch.sum(torch.stack(gaussian_terms))

        return nll

    def call_nll_numpy(self, params):
        if self.dataset == None:
            raise ValueError('Dataset should be initialized before call the NLL!')
        for val, par in zip(params, self.__free_param_list.values()):
            par.data.fill_(val)
        NLL = self.call_nll()
        return NLL.cpu().detach().numpy()

    def call_grad_numpy(self, params):
        if self.dataset == None:
            raise ValueError('Dataset should be initialized before call the NLL!')
        for val, par in zip(params, self.__free_param_list.values()):
            par.data.fill_(val)
        NLL = self.call_nll()
        grad = torch.stack(atg.grad(NLL, self.__free_param_list.values()), dim=0)
        return grad.cpu().detach().numpy()

    def update_vals(self, params):
        param_index = 0
        for par in self.__free_param_list.values():
            if par.numel() == 1:  # Scalar parameters
                par.data.fill_(params[param_index])
                param_index += 1
            else:  # 1D vector parameters
                num_elements = par.numel()
                par.data = torch.tensor(params[param_index:param_index+num_elements], dtype=par.dtype).reshape(par.shape)
                param_index += num_elements

    def objective(self, params):
        if self.dataset is None:
            raise ValueError('Dataset should be initialized before calling the NLL!')

        self.update_vals(params)

        # Clear previous gradients
        self.amp.zero_grad(set_to_none=True)

        # Compute gradients
        NLL = self.call_nll()
        NLL.backward(retain_graph=True)
        grad_flat = torch.cat([par.grad.flatten() for par in self.__free_param_list.values()])
        #grads = atg.grad(NLL, self.__free_param_list.values(), retain_graph=True)
        #grad_flat = torch.cat([g.flatten() for g in grads])

        return NLL.cpu().detach().numpy(), grad_flat.cpu().detach().numpy()
        #return NLL.cpu().detach().numpy()

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
        
        return self.call_nll()

    def hessian(self):
        params = tuple(self.__free_param_list.values())
        flat_params = torch.cat([p.flatten() for p in params])
    
        def flat_func(flat_args):
            args = []
            idx = 0
            for p in params:
                args.append(flat_args[idx:idx+p.numel()].reshape(p.shape))
                idx += p.numel()
            return self.func_call(*args)
    
        hessian = atg.functional.hessian(flat_func, flat_params)
        return hessian

    def inverse_hessian(self):
        hess = self.hessian()
        try:
            # Calculate the condition number
            cond = torch.linalg.cond(hess)
            if cond > 1e15:  # Set a threshold
                print(f"Warning: Hessian matrix condition number is large ({cond:.2e}), which may lead to unstable results.")
            return torch.inverse(hess)
        except RuntimeError as e:
            try:
                print(f"Cannot calculate the inverse of Hessian matrix {e}, try to calculate the pseudo-inverse.")
                return torch.linalg.pinv(hess)
            except RuntimeError as e:
                print(f"Cannot calculate the pseudo-inverse of Hessian matrix: {e}")
                epsilon = 1e-6 # Can be adjusted as needed
                try:
                    return torch.inverse(hess + epsilon * torch.eye(hess.shape[0]))
                except RuntimeError as e:
                    print(f"Cannot calculate the inverse of Hessian matrix: {e}")
                    return torch.full_like(hess, float('nan'))
        
    def std_error(self):
        try:
            inv_hess = self.inverse_hessian()
            std_err = torch.sqrt(torch.diag(inv_hess))
            return std_err, inv_hess
        except RuntimeError:
            print("Warning: Hessian matrix is not invertible, cannot calculate standard errors.")
            hess = self.hessian()
            return torch.full((hess.shape[0],), float('nan')), torch.full_like(hess, float('nan'))

    @property 
    def free_param_list(self) -> dict:
        return self.__free_param_list

class GaussianNLL(NegativeLogLikelihood):
    def __init__(self, amp: nn.Module, data: TensorDict):
        super().__init__(amp, data)
        self.loss = nn.GaussianNLLLoss(reduction='sum', eps=0)

    def call_nll(self):
        nll = super().call_nll()

        # If amp values contain NaN or inf, raise a ValueError exception
        #amp_val = self.amp(self.dataset)
        #if torch.isnan(amp_val).any() or torch.isinf(amp_val).any():
        #    raise ValueError("Amp_val contains nan or inf values.")
        nll = nll + torch.sum((self.dataset['photon_electron']-self.amp_val)**2/self.dataset['pe_uncertainty']**2)
        #nll = nll + torch.sum((self.dataset['photon_electron']-sel)**2/self.dataset['pe_uncertainty']**2)
        return nll

    def call_nll_nosig(self):
        # Only for Chopped design. Assuming that all the backgrounds are shot noises.
        return torch.sum((self.dataset['photon_electron'])**2/self.dataset['pe_uncertainty']**2).cpu().detach().numpy()

class PoissonNLL(NegativeLogLikelihood):
    def __init__(self, amp: nn.Module, data: TensorDict):
        super().__init__(amp, data)
        self.loss = nn.PoissonNLLLoss()

    def call_nll(self):
        # Poisson distribution
        #nll = super().call_nll() + self.loss(self.amp(self.dataset), self.dataset['photon_electron'])
        #predicted_value = self.amp(self.dataset)
        #nll = super().call_nll() + torch.sum(predicted_value - self.dataset['photon_electron'] * torch.log(predicted_value))
        nll = super().call_nll() + torch.sum(self.amp_val - self.dataset['photon_electron'] * torch.log(self.amp_val))
        return nll