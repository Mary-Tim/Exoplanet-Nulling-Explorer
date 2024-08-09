import torch
import torch.nn as nn

import os, h5py
import numpy as np
from datetime import datetime

from nullingexplorer.fitter import ENEFitter
from nullingexplorer.generator import ObservationCreator, AmplitudeCreator
from nullingexplorer.io import DataHandler

class ToyMonteCarlo():
    def __init__(self, path = None) -> None:
        self.__result = {
            "param_name": None,
            "bkg_nll"   : [],
            "best_nll"   : [],
            "true_val" : [],
            "fitted_val" : [],
            "std_err"   : [],
        }
        self.__param_name = None
        self.__obs_creator = ObservationCreator()
        self.__data_handler = DataHandler()

        start_time = datetime.now()
        if path is None:
            self.__path = f"results/Toy_{start_time.strftime('%Y%m%d_%H%M%S')}"
        else:
            self.__path = f"{path}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(self.__path):
            os.mkdir(self.__path)
        print(f"Save the result to: {self.__path}")

    def do_a_toy(self, gen_amp_config, fit_amp_config, obs_config, random_fit_number=50, save_toy_result=False, *args, **kwargs):
        # Generate toy MC
        print("do_a_toy(): Generate toy MC ......")
        gen_amp = AmplitudeCreator(config = gen_amp_config)
        self.__obs_creator.load(config=obs_config)
        gen_data = self.__obs_creator.generate()
        gen_data['photon_electron'] = torch.poisson(gen_amp(gen_data))
        self.__data_handler.data = gen_data

        data = self.__data_handler.diff_data(obs_creator=self.__obs_creator)

        # Fit toy MC
        print("do_a_toy(): Fit toy MC ......")
        fitter = ENEFitter(amp = AmplitudeCreator(config=fit_amp_config), data=data, auto_save=save_toy_result, output_path=self.__path)
        #for planet in fit_amp_config['Amplitude']:
        #    fitter.search_planet(planet, random_number=random_fit_number)
        result = fitter.fit_all(if_random=True, if_std_err=True, random_number=random_fit_number)
        if save_toy_result:
            fitter.fit_result.draw_scan_result(*args, **kwargs)

        # Save result
        if self.__result["param_name"] is None:
            self.__result["param_name"] = result["param_name"]
        true_val = np.array([val.item() for __, val in gen_amp.named_parameters()])
        self.__result["true_val"].append(true_val)
        self.__result["bkg_nll"].append(fitter.NLL.call_nll_nosig())
        self.__result["best_nll"].append(result["best_nll"])
        self.__result["fitted_val"].append(result["param_val"])
        self.__result["std_err"].append(result["std_err"])

        # Clean GPU memory
        torch.cuda.empty_cache()

    def save_all(self):

        with h5py.File(f"{self.__path}/toy_MC_result.hdf5", 'w') as file:
            for key, val in self.__result.items():
                if key != "param_name":
                    val = np.array(val)
                #print(f"Save {key}: {val}")
                file.create_dataset(key, data=val)