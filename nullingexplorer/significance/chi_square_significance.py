import torch
import torch.nn as nn

import copy, os
import h5py
import numpy as np
from tensordict import TensorDict
from datetime import datetime

from nullingexplorer.generator import ObservationCreator, AmplitudeCreator
from nullingexplorer.io import DataHandler
from nullingexplorer.fitter import ENEFitter

class ChiSquareSignificance():
    def __init__(self, planet: str, obs_config: dict, gen_amp_config: dict, fit_amp_config: dict) -> None:
        self.__obs_config = obs_config
        self.__gen_amp_config = gen_amp_config
        self.__fit_amp_config = fit_amp_config
        self.__no_other_planet = False

        self.__obs_creator = ObservationCreator()
        self.__obs_creator.load(self.__obs_config)
        self.__data_handler = DataHandler()

        if not self.__fit_amp_config['Amplitude'].get(planet):
            raise KeyError(f"{planet} is not in the fitting amplitude.")
        if self.__gen_amp_config['Amplitude'].get(planet):
            raise KeyError(f"{planet} should not be in the generation amplitude.")
        
        self.__bkg_amp_config = copy.deepcopy(self.__fit_amp_config)
        del self.__bkg_amp_config['Amplitude'][planet]
        if self.__bkg_amp_config['Amplitude'] == {}:
            self.__no_other_planet = True

    def generate_toy_mc(self, number = None, save_mc=False, *args, **kwargs):
        amp = AmplitudeCreator(config=self.__gen_amp_config)
        data = self.__obs_creator.generate()
        data['photon_electron'] = torch.poisson(amp(data))
        self.__data_handler.data = data

        if save_mc:
            self.__data_handler.save(f"{self.__output_path}/toy_{number}.hdf5")

        return self.__data_handler.diff_data(obs_creator=self.__obs_creator)
    
    def toy_fit(self, data, random_fit_number=50, *args, **kwargs):
        sig_fitter = ENEFitter(amp=AmplitudeCreator(config=self.__fit_amp_config), data=data, output_path=self.__output_path, check_boundary=False, *args, **kwargs)
        print("toy_fit(): Fitting signal PDF ......")
        sig_nll = sig_fitter.fit_all(if_random=True, if_std_err=False, random_number=random_fit_number)['best_nll']
        if self.__no_other_planet:
            bkg_nll = sig_fitter.NLL.call_nll_nosig()
        else:
            bkg_fitter = ENEFitter(amp=AmplitudeCreator(config=self.__bkg_amp_config), data=data, check_boundary=False, *args, **kwargs)
            print("toy_fit(): Fitting background PDF ......")
            bkg_nll = bkg_fitter.fit_all(if_random=True, if_std_err=False, random_number=random_fit_number)['best_nll']

        return sig_nll, bkg_nll

    def pseudoexperiments(self, number_of_toy_mc:int, path:dict = None, save = True, *args, **kwargs):
        if save:
            start_time = datetime.now()
            self.__output_path = f"results/Signi_{start_time.strftime('%Y%m%d_%H%M%S')}"
            if not os.path.exists(self.__output_path):
                os.makedirs(self.__output_path)

        sig_nll_array = np.zeros(number_of_toy_mc)
        bkg_nll_array = np.zeros(number_of_toy_mc)
        for i in range(number_of_toy_mc):
            print(f"Processing toy MC {i+1} ...... {number_of_toy_mc} in total.")
            data = self.generate_toy_mc(number=i, *args, **kwargs)
            sig_nll_array[i], bkg_nll_array[i] = self.toy_fit(data, *args, **kwargs)
            print(f"Toy MC {i+1}: Signal NLL: {sig_nll_array[i]:.3f}, Background NLL: {bkg_nll_array[i]:.3f}")
        torch.cuda.empty_cache()

        if save:
            print(f"Save the result to: {self.__output_path}")
            with h5py.File(f"{self.__output_path}/toy_nll_distribution.hdf5", 'w') as file:
                file.create_dataset("sig_nll_array", data=sig_nll_array)
                file.create_dataset("bkg_nll_array", data=bkg_nll_array)