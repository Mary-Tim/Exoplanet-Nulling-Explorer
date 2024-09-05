import sys
sys.path.append('..')

import numpy as np
np.random.seed(2024)
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import basinhopping
import pandas as pd
import copy, os
import h5py

import torch
torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float64)

from nullingexplorer.significance import PoissonSignificance
from nullingexplorer.utils import Constants as cons

class PpopLoader():
    def __init__(self, csv_path:str):
        self.__i_star = 0
        self.__n_star = 0
        self.__pd_data = pd.read_csv(csv_path, sep='\t')
        #print(self.__pd_data)
        self.__n_star = self.__pd_data['Nstar'].max() + 1
        print(f"Number of Star: {self.__n_star}")

    def next_star(self):
        if self.__i_star >= self.__n_star:
            print("Reach the end of P-pop list.")
            return None
        
        stellar_info = self.get_star(i_star = self.__i_star)
        self.__i_star += 1
        while(len(stellar_info) == 0):
            stellar_info = self.get_star(i_star = self.__i_star)
            self.__i_star += 1

        return stellar_info
    
    def set_param_dict(self, dict):
        self.__pd_data = self.__pd_data[dict]

    def get_star(self, i_star) -> pd.DataFrame :
        if i_star >= self.__n_star:
            raise KeyError(f"Star {i_star} not in the P-pop list.")
        
        return self.__pd_data[self.__pd_data['Nstar']==i_star].reset_index()

    def numpy_array(self, i_star = None) -> np.ndarray:
        if i_star is None:
            return self.get_star(self.__i_star).to_numpy()
        else:
            return self.get_star(i_star).to_numpy()

    @property
    def n_star(self):
        return self.__n_star
    
    @property
    def i_star(self):
        return self.__i_star

    @property
    def ppop_data(self):
        return self.__pd_data
    
    @i_star.setter
    def i_star(self, i_star):
        self.__i_star = i_star

class ObservationPlanner():
    '''
    P-pop parameters needed by this class:
        'Rp',       # Planet radius                         [R_earth]
        'AngSep',   # Planet projected angular separation   [arcsec]
        'Tp',       # Planet equilibrium temperature        [K]
        'Nstar',    # Number of the star
        'z',        # Exozodiacal dust level
        'Rs',       # Host star radius                      [R_sun]
        'Ts',       # Host star effective temperature       [K]
        'Ds',       # Host star distance                    [pc]
        'RA',       # Host star right ascension             [deg]
        'Dec'       # Host star declination                 [deg]
    '''
    def __init__(self, loader: PpopLoader):
        self.__ppop_loader = loader
        self.__poisson_sigfi = PoissonSignificance()
        self.__param_list = [
            'Rp',       # Planet radius                         [R_earth]
            'AngSep',   # Planet projected angular separation   [arcsec]
            'Tp',       # Planet equilibrium temperature        [K]
            'Nstar',    # Number of the star
            'z',        # Exozodiacal dust level
            'Rs',       # Host star radius                      [R_sun]
            'Ts',       # Host star effective temperature       [K]
            'Ds',       # Host star distance                    [pc]
            'RA',       # Host star right ascension             [deg]
            'Dec'       # Host star declination                 [deg]
        ]
        self.__ppop_loader.set_param_dict(self.__param_list)
        self.__obs_template = {
            'Spectrum':{
                'Type': 'Resolution',
                'R': 20,
                'Low': 5.,
                'High': 17.,        # unit: micrometer
            },
            'Observation':{
                'ObsNumber': 360,
                'IntegrationTime': 1,  # unit: second
                'ObsMode': [1],  # [1] or [-1] or [1, -1]
                'Phase':{
                    'Start' : 0.,
                    'Stop': 360.,   # unit: degree
                },
                'Baseline':{
                    'Type': 'Constant',
                    'Value': 10.,  # unit: meter
                },
            },
            'Configuration':{
                # Formation parameters
                'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
                'formation_longitude': 0.,  # Formation longitude [degree] 
                'formation_latitude' : 0.,  # Formation latitude [degree] 
                # Instrument parameters
                'mirror_diameter': 3.5,   # Diameter of MiYin primary mirror [meter]
                'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
                'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
                'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
            }
        }

        self.__planet_template = {
            'Model': 'PlanetBlackBody',
            'Spectrum': 'InterpBlackBody',
            'Parameters':
            {
                'radius':         {'mean': 6371.e3},
                'temperature':    {'mean': 285.},
                'ra':            {'mean': 100.},
                'dec':            {'mean': 0.},
            },
        }
        self.__sig_template = {
            'Amplitude':{
            },
            'Instrument': 'MiYinBasicType',
            'TransmissionMap': 'DualChoppedDestructive',
            'Configuration':{
                'distance': 10,         # distance between Miyin and target [pc]
            }
        }

        self.__bkg_template = {
            'Amplitude':{
                'star':{
                    'Model': 'StarBlackBodyConstant',
                    'Buffers': {
                        'vol_number': 50,
                        'wl_number': 2   
                    }
                },
                'local_zodi':{
                    'Model': 'LocalZodiacalDustConstant',
                    'Buffers': {
                        'vol_number': 50,
                    }
                },
                'exo_zodi':{
                    "Model": 'ExoZodiacalDustConstant',
                    'Buffers': {
                        'vol_number': 50,
                        'wl_number': 2   
                    }
                },
            },
            'Instrument': 'MiYinBasicType',
            'TransmissionMap': 'DualChoppedDestructive',
            #'Electronics': {
            #    'Model': 'UniformElectronics',
            #    'Buffers': {
            #        'noise_rate': 10.0,
            #    },
            #},
            'Configuration':{
                'distance': 10,         # distance between Miyin and target [pc]
                'star_radius': 695500,  # Star radius [kilometer]
                'star_temperature': 5772,   # Star temperature [Kelvin]
                'target_longitude': 0.,     # Ecliptic longitude [degree]
                'target_latitude': 0.,      # Ecliptic latitude  [degree]
                'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
            }
        }

    def get_baseline(self, distance, angular):
        '''
        a and b obtained from interpolation
        k = a * distance + b
        baseline = k * angular
        '''
        a = -14.26165496
        b = 1810.93850806
        k = a * distance + b
        baseline = k / angular
        #if baseline < 30.:
        #    baseline = 30.

        return baseline

    def habitable_zone(self, radius, temperature, distance):
        sun_radius = 695500
        sun_temperature = 5780
        habitable_parameters = np.array(
            [
                [1.7763,    1.4335e-4,  3.3954e-9,  -7.6364e-12,    -1.1950e-15],
                [0.3207,    5.4471e-5,  1.5275e-9,  -2.1709e-12,    -3.8282e-16]
            ],
            dtype=np.float32
        )

        relative_radius = radius / sun_radius
        relative_temperature = temperature / sun_temperature
        relative_lumi = relative_radius ** 2 * relative_temperature ** 4

        hz_temperature = temperature - sun_temperature

        s_eff_in = habitable_parameters[0, 0] + \
            habitable_parameters[0, 1] * hz_temperature + \
            habitable_parameters[0, 2] * hz_temperature ** 2 + \
            habitable_parameters[0, 3] * hz_temperature ** 3 + \
            habitable_parameters[0, 4] * hz_temperature ** 4

        s_eff_out = habitable_parameters[1, 0] + \
            habitable_parameters[1, 1] * hz_temperature + \
            habitable_parameters[1, 2] * hz_temperature ** 2 + \
            habitable_parameters[1, 3] * hz_temperature ** 3 + \
            habitable_parameters[1, 4] * hz_temperature ** 4

        #hz_in = np.sqrt(relative_lumi/s_eff_in) 
        #hz_out = np.sqrt(relative_lumi/s_eff_out) 
        hz_in = np.sqrt(relative_lumi/s_eff_in) * cons._au_to_meter / (distance * cons._pc_to_meter) * cons._radian_to_mas
        hz_out = np.sqrt(relative_lumi/s_eff_out) * cons._au_to_meter / (distance * cons._pc_to_meter) * cons._radian_to_mas

        return hz_in, hz_out

    def generate_obs_plan(self, stellar_info: pd.DataFrame):
        #stellar_info = self.__ppop_loader.next_star()
        # observation config
        obs_config = copy.deepcopy(self.__obs_template)
        obs_config['Configuration']['formation_longitude'] = stellar_info['RA'][0]
        # background config
        bkg_config = copy.deepcopy(self.__bkg_template)
        bkg_config['Configuration']['distance'] = stellar_info['Ds'][0]
        bkg_config['Configuration']['star_radius'] = stellar_info['Rs'][0] * cons._sun_radius
        bkg_config['Configuration']['star_temperature'] = stellar_info['Ts'][0]
        bkg_config['Configuration']['target_longitude'] = stellar_info['RA'][0]
        bkg_config['Configuration']['target_latitude'] = stellar_info['Dec'][0]
        bkg_config['Configuration']['zodi_level'] = 3.
        #bkg_config['Configuration']['zodi_level'] = stellar_info['z'][0]

        hz_in, _ = self.habitable_zone(stellar_info['Rs'][0] * cons._sun_radius, stellar_info['Ts'][0], stellar_info['Ds'][0])
        planet_config = copy.deepcopy(self.__planet_template)
        planet_config['Parameters']['ra']['mean'] = hz_in
        #planet_config['Parameters']['ra']['mean'] = self.__planet_template['Parameters']['ra']['mean'] * (10. / stellar_info['Ds'][0])

        obs_config['Observation']['Baseline']['Value'] = self.get_baseline(stellar_info['Ds'][0], planet_config['Parameters']['ra']['mean'])

        #bkg_config['Amplitude'].update(planet_dict)
        sig_config = copy.deepcopy(self.__sig_template)
        sig_config['Amplitude']['planet'] = planet_config
        sig_config['Configuration']['distance'] = stellar_info['Ds'][0]

        return obs_config, bkg_config, sig_config

    def evaluate_stellar_system(self, stellar_info: pd.DataFrame):
        obs_config, bkg_config, planet_config = self.generate_obs_plan(stellar_info)
        self.__poisson_sigfi.obs_config = obs_config

        #print(planet_config)
        sig_pe = self.__poisson_sigfi.gen_sig_pe(sig_config=planet_config)
        bkg_pe = self.__poisson_sigfi.gen_bkg_pe(bkg_config=bkg_config)

        def significance(t):
            return (self.__poisson_sigfi.get_significance(sig_pe=sig_pe*t, bkg_pe=bkg_pe*t)).cpu().detach().numpy()

        def loss(param):
            return np.fabs(7 - significance(param[0]))

        result = basinhopping(loss, 
                    x0=100., 
                    minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': [(0., 200000.)], 'jac': False, 
                                        'options': {'maxcor': 1000, 'ftol': 1e-10, 'maxiter': 100000, 'maxls': 50}}, 
                    stepsize=10,
                    niter=10000,
                    niter_success=50)

        return obs_config['Observation']['Baseline']['Value'], result.x[0], significance(result.x[0]), planet_config['Amplitude']['planet']['Parameters']['ra']['mean']

    def evaluate_all(self, output_path = None):
        self.__ppop_loader.i_star = 0
        result_list = []
        for i in tqdm(range(self.__ppop_loader.n_star)):
            stellar_info = self.__ppop_loader.next_star()
            if stellar_info is None:
                break
            bl, ot, sig, hz_in = self.evaluate_stellar_system(stellar_info)
            stellar_info['Baseline'] = bl
            stellar_info['ObsTime'] = ot
            stellar_info['HZ_in'] = hz_in
            print(f"Distrance:{stellar_info['Ds'][0]}, Baseline: {bl:.3f}, ObsTime: {ot:.3f}, Significance: {sig:.3f}: HZ-in: {hz_in:.3f}")
            result_list.append(stellar_info[0:1])
            #result_list.append(np.column_stack((stellar_info.to_numpy(), significance, true_ang_sep)))

        #result_numpy = np.vstack(result_list)
        result = pd.concat(result_list)

        start_time = datetime.now()
        if output_path is None:
            output_path = f"results/ObsTime_{start_time.strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        self.__param_list.append("Baseline")
        self.__param_list.append("ObsTime")

        print(f"Save the result to: {output_path}")
        with h5py.File(f"{output_path}/toy_nll_distribution.hdf5", 'w') as file:
            for key in self.__param_list:
                file.create_dataset(key, data=result[key].to_numpy())
            #for i, key in enumerate(self.__param_list):
            #    file.create_dataset(key, data=result_numpy[:,i])

def main():
    # Set device during generation
    torch.set_default_device('cuda:0')
    torch.set_default_dtype(torch.float64)
    #torch.multiprocessing.set_start_method('spawn')

    loader = PpopLoader('MeaYinPlanetPopulation_Full.txt')

    observer = ObservationPlanner(loader)
    observer.evaluate_all()


    #significance = observer.evaluate_stellar_system(stellar_info)
    #print(significance)

if __name__ == '__main__':
    main()