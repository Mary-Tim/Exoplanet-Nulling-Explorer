import torch
import torch.nn as nn

import copy
import numpy as np
from tensordict import TensorDict

from nullingexplorer.generator import ObservationCreator, AmplitudeCreator
from nullingexplorer.io import DataHandler
from nullingexplorer.fitter import ENEFitter

class ChiSquareSignificance():
    def __init__(self, planet: str, obs_config: dict, gen_amp_config: dict, fit_amp_config: dict) -> None:
        self.__planet = planet
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

    def generate_toy_mc(self, number = None, path: str = None):
        amp = AmplitudeCreator(config=self.__gen_amp_config)
        data = self.__obs_creator.generate()
        data['photon_electron'] = torch.poisson(amp(data))
        self.__data_handler.data = data

        if path is not None:
            self.__data_handler.save(f"{path}/toy_{number}.hdf5")

        return self.__data_handler.diff_data(obs_creator=self.__obs_creator)
    
    def toy_fit(self, data):
        sig_fitter = ENEFitter(amp=AmplitudeCreator(config=self.__fit_amp_config), data=data)
        sig_nll = sig_fitter.fit_all(if_random=True, random_number=50)
        if self.__no_other_planet:
            bkg_nll = sig_fitter.NLL.call_nll_nosig()
        else:
            bkg_fitter = ENEFitter(amp=AmplitudeCreator(config=self.__bkg_amp_config), data=data)
            bkg_nll = bkg_fitter.fit_all(if_random=True, random_number=50)

        return sig_nll, bkg_nll

    def pseudoexperiments(self, number_of_toy_mc:int, mc_path:dict = None):
        sig_nll_array = np.zeros(number_of_toy_mc)
        bkg_nll_array = np.zeros(number_of_toy_mc)
        for i in range(number_of_toy_mc):
            print(f"Processing toy MC {i}......")
            data = self.generate_toy_mc(number=i, path=mc_path)
            sig_nll_array[i], bkg_nll_array[i] = self.toy_fit(data)
            print(f"Toy MC {i}: Signal NLL: {sig_nll_array[i]:.3f}, Background NLL: {bkg_nll_array[i]:.3f}")
        torch.cuda.empty_cache()


    