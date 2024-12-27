import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as torchmp

import copy
import os
import numpy as np
from scipy.optimize import basinhopping, minimize
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
    def __init__(self, amp: BaseAmplitude, data: TensorDict, NLL_type = 'gaussian', min_method='L-BFGS-B', check_boundary = True, multi_gpu = False, *args, **kwargs):
        self.amp = amp
        self.data = data.detach()
        if NLL_type not in ENEFitter.NLL_dict.keys():
            raise KeyError(f"NLL type {NLL_type} not found!")
        self.__NLL_type = NLL_type
        self.NLL = ENEFitter.NLL_dict[self.__NLL_type](self.amp, self.data)
        self.fit_result = FitResult(*args, **kwargs)
        self.min_method = min_method
        self.check_boundary = check_boundary

        self.__n_gpus = torch.cuda.device_count()
        self.__multi_gpu = multi_gpu

    @staticmethod
    def scipy_minimize(NLL, init_val=None, min_method='L-BFGS-B', *args, **kwargs):
        boundary = NLL.get_boundaries()
        boundary_lo = np.array([bound[0] for bound in boundary])
        boundary_hi = np.array([bound[1] for bound in boundary])
        if init_val is None:
            init_val = np.random.uniform(low=boundary_lo, high=boundary_hi)
        result = minimize(NLL.objective, 
                    x0=init_val, 
                    method = min_method,
                    jac = True,
                    bounds = boundary,
                    options={'maxcor': 100, 'ftol': 1e-15, 'maxiter': 10000, 'maxls': 50})
        return result

    @staticmethod
    def scipy_basinhopping(NLL, stepsize=0.25, niter=10000, niter_success=20, maxls=50, init_val=None, min_method='L-BFGS-B', *args, **kargs):
        boundary = NLL.get_boundaries()
        boundary_lo = np.array([bound[0] for bound in boundary])
        boundary_hi = np.array([bound[1] for bound in boundary])
        if init_val is None:
            init_val = np.random.uniform(low=boundary_lo, high=boundary_hi)
        result = basinhopping(NLL.objective, 
                    x0=init_val, 
                    #minimizer_kwargs={'method': min_method, 'bounds': boundary, 'jac': False, 
                    minimizer_kwargs={'method': min_method, 'bounds': boundary, 'jac': True, 
                                        'options': {'maxcor': 100, 'ftol': 1e-10, 'maxiter': 10000, 'maxls': maxls}}, 
                    T=2.5,
                    stepsize=stepsize,
                    niter=niter,
                    niter_success=niter_success)
        return result

    def a_search(self, NLL, iter_number, fit_params=[], *args, **kwargs):
        result = None
        scan_nll = np.zeros(iter_number)
        # Initialize array to record all attempted parameter values based on the length of fit_params
        if len(fit_params) != 0:
            NLL.config_fit_params(fit_params)
            #scan_result = np.zeros((iter_number, len(fit_params)))
        #else:
        scan_result = np.zeros((iter_number, NLL.num_of_params))
        boundary = NLL.get_boundaries()
        # For each attempt of the random search 
        for i in tqdm(range(iter_number)):
            flag = False                                      
            retry_times = 0
            # Continue attempting until a successful minimization result is found
            while(not flag):                                  
                try:
                    this_result = ENEFitter.scipy_basinhopping(NLL=NLL, min_method=self.min_method, *args, **kwargs)
                except ValueError:
                    print(f"Detect ValueError during fitting! Current param values: {self.NLL.get_param_values()}")
                    continue
                except RuntimeError:
                    print(f"Detect RuntimeError during fitting! Current param values: {self.NLL.get_param_values()}")
                    continue

                #this_result = ENEFitter.scipy_basinhopping(NLL=NLL, min_method=self.min_method, *args, **kwargs)

                flag = this_result.success
                if flag == False:
                    retry_times += 1
                if(retry_times > 100):
                    print("All retry fails! Move to the next point.")
                    break
            # Check if any parameter reach to the boundary
            if self.check_boundary:
                reach_to_boundary = False
                for val, bound in zip(this_result.x, boundary):
                    if np.fabs(val - bound[0])<1e-3 or np.fabs(val - bound[1])<1e-3:
                        reach_to_boundary = True
                        break
                if reach_to_boundary:
                    continue
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
        index = np.nonzero(scan_nll)
        return result, scan_nll[index], scan_result[index]

    def one_process(self, rank, NLL_list, iter_number, fit_params, *args, **kwargs):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=self.__n_gpus)

        NLL_list[rank].to(rank)
        NLL_list[rank].amp.to(rank)
        NLL_list[rank].dataset = NLL_list[rank].dataset.to(rank)
        result, scan_nll, scan_result = self.a_search(NLL=NLL_list[rank], iter_number=iter_number, fit_params=fit_params, *args, **kwargs)

        #print(f"Scan_NLL from cuda:{rank}: {scan_nll}")
        self.result_queue.put((result, scan_nll, scan_result))
        print(f"Save cuda:{rank} result to queue.")

        dist.destroy_process_group()

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

        if self.__multi_gpu == False:
            result, scan_nll, scan_result = self.a_search(self.NLL, iter_number=random_number, fit_params=fit_params, *args, **kwargs)
            self.fit_result.set_item('scan_nll', scan_nll)
            self.fit_result.set_item('scan_result', scan_result)
        else:
            if self.__n_gpus < 2:
                raise ValueError("Number of avaiable CUDA driver less than 2, Please disable MultiGPU mode!")

            # Multiprocessing
            NLL_list = [ENEFitter.NLL_dict[self.__NLL_type](copy.deepcopy(self.amp), self.data) for _ in range(self.__n_gpus)]

            iter_number = random_number // self.__n_gpus
            if random_number % self.__n_gpus > 0:
                iter_number += 1

            result_dict = {
                'result'    : [None] * self.__n_gpus,
                'scan_nll'  : [None] * self.__n_gpus,
                'scan_result': [None] * self.__n_gpus
            }
            self.result_queue = torchmp.JoinableQueue()
            ctx = torchmp.spawn(self.one_process, args=(NLL_list, iter_number, fit_params), nprocs=self.__n_gpus, join=False)

            # Read results from subprocesses
            for i in range(self.__n_gpus):
                temp_result = self.result_queue.get()
                result_dict['result'][i] = copy.deepcopy(temp_result[0])
                result_dict['scan_nll'][i] = copy.deepcopy(temp_result[1])
                result_dict['scan_result'][i] = copy.deepcopy(temp_result[2])
            self.result_queue.task_done()

            # End of Multiprocesses
            ctx.join()

            # Find best result and save all
            result = None
            for re in result_dict['result']:
                if re is not None:
                    if result is None:
                        result = re
                    else:
                        if re.fun < result.fun:
                            result = re
            self.fit_result.set_item('scan_nll', np.hstack(result_dict['scan_nll']))
            self.fit_result.set_item('scan_result', np.vstack(result_dict['scan_result']))

        return result

    def scan_search(self, scan_precision = 100, scan_params = [], *args, **kwargs):
        if len(scan_params) != 2:
            raise ValueError(f"Required 2 scan parameters, {len(scan_params)} inputted.")

        self.NLL.fix_param(scan_params)

        x_boundary = self.NLL.get_boundary(scan_params[0])
        y_boundary = self.NLL.get_boundary(scan_params[1])

        x_list = torch.linspace(x_boundary[0], x_boundary[1], scan_precision)
        y_list = torch.linspace(y_boundary[0], y_boundary[1], scan_precision)
        x_grid, y_grid = torch.meshgrid(x_list, y_list, indexing='ij')

        points = torch.stack((x_grid.flatten(), y_grid.flatten()), -1)
        nll_grid = np.zeros(len(points), dtype=np.float64)

        result = None
        best_point = None
        for i, point in tqdm(enumerate(points), total=len(points)):
            flag = False
            retry_times = 0
            self.NLL.set_param_val(scan_params[0], point[0], constant=True)
            self.NLL.set_param_val(scan_params[1], point[1], constant=True)
            while(not flag):                                  
                this_result = ENEFitter.scipy_minimize(self.NLL)
                flag = this_result.success
                if flag == False:
                    #print(f"Fail to find the minimum result, retry. NLL: {this_result.fun}")
                    retry_times += 1
                if(retry_times > 1000):
                    print("All retry fails! Move to the next point.")
                    break
            nll_grid[i] = this_result.fun
        if result == None:
            result = this_result
            best_point = point
        elif this_result.fun < result.fun:
            result = this_result
            best_point = point

        self.fit_result.draw_meshgrid_result(   x_grid.cpu().detach().numpy(), 
                                                y_grid.cpu().detach().numpy(), 
                                                nll_grid.reshape(scan_precision, scan_precision).cpu().detach().numpy())

        return result, best_point

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
        result = ENEFitter.scipy_basinhopping(self.NLL, min_method=self.min_method, init_val=init_val, *args, **kwargs)
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
        if result is None:
            print(f"Planet {amp_name} not found! return None.")
            return None

        result = self.precision_search(fit_params=planet_params, init_val=list(result.x), 
                                       stepsize=0.1, niter=10000, niter_success=500, *args, **kwargs)
        #for i, name in enumerate(planet_params):
        #    self.NLL.set_param_val(name, result.x[i])
        self.NLL.update_vals(result.x)
        self.fit_result.load_fit_result(self.NLL, result)
        if std_err == True:
            self.fit_result.evaluate_std_error()
        self.fit_result.print_result()

        if draw == True:
            self.fit_result.draw_scan_result(file_name=f"{amp_name}", show=show, *args, **kwargs)

        self.fit_result.save(name=amp_name)
        return result

    def fit_all(self, if_random=False, if_std_err=True, *args, **kwargs):
        print("Fitting all parameters...")
        self.NLL.free_all_params()
        if if_random:
            result = self.random_search(*args, **kwargs)
            if result is None:
                print("No result found! return None.")
                return None
            else:
                result = self.precision_search(init_val=list(result.x), stepsize=0.1, niter=10000, 
                                            niter_success=500, *args, **kwargs)
        else:
            result = self.precision_search(stepsize=0.1, niter=10000, niter_success=500, *args, **kwargs)
        self.fit_result.load_fit_result(self.NLL, result)
        if if_std_err:
            self.fit_result.evaluate_std_error()
        self.fit_result.print_result()
        self.fit_result.save(name="all")

        return self.fit_result.result