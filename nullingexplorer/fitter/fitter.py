import torch
import torch.nn as nn
import torch.autograd as atg
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

import numpy as np
from scipy.optimize import basinhopping
from tqdm import tqdm
from tensordict import TensorDict

from nullingexplorer.utils import Constants as cons
from nullingexplorer.fitter import GaussianNLL, PoissonNLL
from nullingexplorer.model.amplitude import BaseAmplitude
from nullingexplorer.io import FitResult


class ENEFitter():
    NLL_dict = {
        'gaussian': GaussianNLL,
        'poisson': PoissonNLL
    }
    def __init__(self, amp: BaseAmplitude, data: TensorDict, NLL_type = 'gaussian', min_method='L-BFGS-B'):
        self.amp = amp
        self.data = data
        if NLL_type not in self.NLL_dict.keys():
            raise KeyError(f"NLL type {NLL_type} not found!")
        self.NLL = self.NLL_dict[NLL_type](self.amp, self.data)
        self.fit_result = FitResult(auto_save=False)
        self.min_method = min_method

    def scipy_basinhopping(self, stepsize=1., niter=1000, niter_success=50, init_val=None, *args, **kargs):
        boundary = self.NLL.get_boundary()
        boundary_lo = np.array([bound[0] for bound in boundary])
        boundary_hi = np.array([bound[1] for bound in boundary])
        if init_val is None:
            init_val = np.random.uniform(low=boundary_lo, high=boundary_hi)
        result = basinhopping(self.NLL.objective, 
                    x0=init_val, 
                    minimizer_kwargs={'method':self.min_method, 'bounds': boundary, 'jac': True, 
                                      'options': {'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50}}, 
                    stepsize=stepsize,
                    niter=niter,
                    niter_success=niter_success)
        return result

    def random_search(self, random_number = 50, fit_params = [], *args, **kwargs):
        """
        Utilizes a random search method to find the parameter combination with the minimum negative log-likelihood value.
        
        Parameters:
        random_number: int, the number of times to perform the random search.
        fit_params: list, the initial list of fitting parameters.
        *args, **kwargs: arguments passed to the scipy_basinhopping function.
        
        Returns:
        The final optimization result.
        """
        print("Random scanning...")
        # Initialize result variable and array to record all attempted negative log-likelihood values
        result = None
        scan_nll = np.zeros(random_number)
        # Initialize array to record all attempted parameter values based on the length of fit_params
        if len(fit_params) != 0:
            self.NLL.config_fit_params(fit_params)
            scan_result = np.zeros((random_number, len(fit_params)))
        else:
            scan_result = np.zeros((random_number, self.NLL.num_of_params))
        # For each attempt of the random search 
        for i in tqdm(range(random_number)):
            flag = False                                      
            retry_times = 0
            # Continue attempting until a successful minimization result is found
            while(not flag):                                  
                this_result = self.scipy_basinhopping(*args, **kwargs)
                flag = this_result.success
                if flag == False:
                    retry_times += 1
                if(retry_times > 100):
                    print("All retry fails! Move to the next point.")
                    break
            # Record the negative log-likelihood value and parameter values of this attempt
            scan_nll[i] = this_result.fun
            for j in range(len(this_result.x)):
                scan_result[i, j] = this_result.x[j]
            # Update the best result
            if result == None:
                result = this_result
            elif this_result.fun < result.fun:
                result = this_result
        # Store search results in fit_result
        self.fit_result.set_item('scan_nll', scan_nll)
        self.fit_result.set_item('scan_result', scan_result)

        return result

    def precision_search(self, init_val=None, fit_params = [], *args, **kwargs):
        """
        Performs a precise search using the Basin Hopping algorithm.

        This method first checks if fit parameters are provided. If so, it configures these parameters. Then, it utilizes the Basin Hopping algorithm from scipy to find a minimum.

        Parameters:
        fit_params: A list containing parameters for fitting. If this parameter is provided, it will be used to configure the fit parameters of the NLL object.
        *args, **kwargs: Arguments passed to the scipy.optimize.basinhopping function. These arguments can control the behavior of the Basin Hopping algorithm.

        Returns:
        The result of executing the scipy.optimize.basinhopping function.
        """
        print("Precision searching...")
        if init_val is None:
            init_val = [val.data.item() for val in self.NLL.free_param_list.values()]
        if len(fit_params) != 0:
            self.NLL.config_fit_params(fit_params)
        result = self.scipy_basinhopping(init_val=init_val, *args, **kwargs)
        return result

    def search_planet(self, amp_name:str, std_err=False, draw=False, show=False, *args, **kwargs):
        print(f"Searching planet {amp_name}...")
        self.NLL.free_all_params()
        name_of_params = self.NLL.name_of_params
        planet_params = []
        for name in name_of_params:
            if name.find(amp_name) != -1:
                planet_params.append(name)
        if len(planet_params) == 0:
            raise KeyError(f"Planet {amp_name} not found!")
        result = self.random_search(fit_params=planet_params, *args, **kwargs)
        result = self.precision_search(fit_params=planet_params, init_val=list(result.x), 
                                       stepsize=0.1, niter=10000, niter_success=500, *args, **kwargs)
        for i, name in enumerate(planet_params):
            self.NLL.set_param_val(name, result.x[i])
        self.fit_result.load_fit_result(self.NLL, result)
        if std_err == True:
            self.fit_result.evaluate_std_error()
        self.fit_result.print_result()

        if draw == True:
            position_name = ['', '']
            for name in planet_params:
                if name.endswith('ra'):
                    position_name[0] = name
                if name.endswith('dec'):
                    position_name[1] = name
            self.fit_result.draw_scan_result(position_name, file_name=f"{amp_name}", show=show)

        return result

    def fit_all(self, if_random=False, *args, **kwargs):
        print("Fitting all parameters...")
        self.NLL.free_all_params()
        if if_random:
            result = self.random_search(*args, **kwargs)
            result = self.precision_search(init_val=list(result.x), stepsize=0.1, niter=10000, 
                                           niter_success=500, *args, **kwargs)
        else:
            result = self.precision_search(stepsize=0.1, niter=10000, niter_success=500, *args, **kwargs)
        self.fit_result.load_fit_result(self.NLL, result)
        self.fit_result.evaluate_std_error()
        self.fit_result.print_result()

        return self.fit_result.result['best_nll']
