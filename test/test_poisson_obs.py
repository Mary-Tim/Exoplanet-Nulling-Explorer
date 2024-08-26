import sys
sys.path.append('..')

import numpy as np
np.random.seed(2024)
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm
from datetime import datetime
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

class PoissonObservation():
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
                'IntegrationTime': 100,  # unit: second
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
                'mirror_diameter': 4,   # Diameter of MiYin primary mirror [meter]
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
                'ra':            {'mean': 62.5},
                'dec':            {'mean': 78.1},
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
                    'Model': 'StarBlackBodyMatrix',
                },
                'local_zodi':{
                    'Model': 'LocalZodiacalDustMatrix',
                },
                'exo_zodi':{
                    "Model": 'ExoZodiacalDustMatrix',
                },
            },
            'Instrument': 'MiYinBasicType',
            'TransmissionMap': 'DualChoppedDestructive',
            'Configuration':{
                'distance': 10,         # distance between Miyin and target [pc]
                'star_radius': 695500,  # Star radius [kilometer]
                'star_temperature': 5772,   # Star temperature [Kelvin]
                'target_longitude': 0.,     # Ecliptic longitude [degree]
                'target_latitude': 0.,      # Ecliptic latitude  [degree]
                'zodi_level': 3,        # scale parameter for exo-zodi [dimensionless]
            }
        }
    
    def generate_planet_position(self, max_ang, orbit_vector):
        cms_theta = np.random.uniform(0., 2*np.pi)
        z_unit = np.array([0., 0., 1.])
        cos_rotate = np.clip(np.dot(orbit_vector, z_unit)/(np.linalg.norm(orbit_vector)*np.linalg.norm(z_unit)), -1., 1.)

        ra = max_ang * cos_rotate * np.cos(cms_theta)
        dec = max_ang * cos_rotate * np.sin(cms_theta)
        return ra, dec

    def generate_orbit_vector(self):
        theta = np.arccos(np.random.uniform(-1.,1.))
        phi = np.random.uniform(0., 2*np.pi)
        return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], dtype=np.float64)

    def generate_planet_amp(self, stellar_info: pd.DataFrame, orbit_vector: np.ndarray):
        planet_dict = {}
        for index, row in stellar_info.iterrows():
            planet_config = copy.deepcopy(self.__planet_template)
            planet_config['Parameters']['radius']['mean'] = row['Rp'] * cons._earth_radius
            planet_config['Parameters']['temperature']['mean'] = row['Tp']
            ra, dec = self.generate_planet_position(max_ang=row['AngSep']*1000, orbit_vector=orbit_vector)
            planet_config['Parameters']['ra']['mean']= ra
            planet_config['Parameters']['dec']['mean']= dec
            print(f"AngSep: {row['AngSep']},\tra: {ra},\tdec: {dec}")

            planet_dict[f'planet{index}'] = planet_config

        return planet_dict

    def generate_obs_plan(self, stellar_info: pd.DataFrame):
        #stellar_info = self.__ppop_loader.next_star()
        # observation config
        obs_config = copy.deepcopy(self.__obs_template)
        obs_config['Configuration']['formation_longitude'] = stellar_info['RA'][0]
        # background config
        bkg_config = copy.deepcopy(self.__bkg_template)
        bkg_config['Configuration']['distance'] = stellar_info['Ds'][0]
        bkg_config['Configuration']['star_radius'] = stellar_info['Rs'][0] * cons._sun_radius
        bkg_config['Configuration']['star_temperature'] = stellar_info['Rs'][0]
        bkg_config['Configuration']['target_longitude'] = stellar_info['RA'][0]
        bkg_config['Configuration']['target_latitude'] = stellar_info['Dec'][0]
        bkg_config['Configuration']['zodi_level'] = stellar_info['z'][0]

        orbit_vector = self.generate_orbit_vector()
        planet_dict = self.generate_planet_amp(stellar_info=stellar_info, orbit_vector=orbit_vector)
        #bkg_config['Amplitude'].update(planet_dict)

        return obs_config, bkg_config, planet_dict

    def evaluate_all(self, output_path = None):
        self.__ppop_loader.i_star = 0
        result_list = []
        for i in tqdm(range(self.__ppop_loader.n_star)):
            stellar_info = self.__ppop_loader.next_star()
            if stellar_info is None:
                break
            significance, true_ang_sep = self.evaluate_stellar_system(stellar_info)
            stellar_info['sigma'] = significance
            stellar_info['true_ang_sep'] = true_ang_sep
            result_list.append(stellar_info)
            #result_list.append(np.column_stack((stellar_info.to_numpy(), significance, true_ang_sep)))

        #result_numpy = np.vstack(result_list)
        result = pd.concat(result_list)

        start_time = datetime.now()
        if output_path is None:
            output_path = f"results/PoissonObs_{start_time.strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        self.__param_list.append("sigma")
        self.__param_list.append("true_ang_sep")

        print(f"Save the result to: {output_path}")
        with h5py.File(f"{output_path}/toy_nll_distribution.hdf5", 'w') as file:
            for key in self.__param_list:
                file.create_dataset(key, data=result[key].to_numpy())
            #for i, key in enumerate(self.__param_list):
            #    file.create_dataset(key, data=result_numpy[:,i])

    def evaluate_stellar_system(self, stellar_info: pd.DataFrame):
        obs_config, bkg_config, planet_dict = self.generate_obs_plan(stellar_info)
        self.__poisson_sigfi.obs_config = obs_config

        significance = np.zeros(len(stellar_info), dtype=np.float64)
        true_ang_sep = np.zeros(len(stellar_info), dtype=np.float64)

        continuum_nbkg = self.__poisson_sigfi.gen_bkg_pe(bkg_config=bkg_config)

        for i, key in enumerate(planet_dict.keys()):
            # Generate background
            if len(planet_dict) > 1:
                planet_bkg = copy.deepcopy(planet_dict)
                planet_bkg_config = copy.deepcopy(self.__sig_template)
                del planet_bkg[key]
                planet_bkg_config['Amplitude'].update(planet_bkg)
                nbkg = continuum_nbkg + self.__poisson_sigfi.gen_bkg_pe(bkg_config=planet_bkg_config)
            else:
                nbkg = continuum_nbkg

            # Generate signal
            sig_config = copy.deepcopy(self.__sig_template)
            sig_config['Configuration']['distance'] = stellar_info['Ds'][0]
            sig_config['Amplitude'] = {key: planet_dict[key]}
            nsig = self.__poisson_sigfi.gen_sig_pe(sig_config=sig_config)

            # Evaluate significance
            significance[i] = self.__poisson_sigfi.get_significance(sig_pe=nsig, bkg_pe=nbkg)

            # Calculate true angular separation
            ra = planet_dict[key]['Parameters']['ra']['mean'] / 1000.
            dec = planet_dict[key]['Parameters']['dec']['mean'] / 1000.
            true_ang_sep[i] = np.sqrt(ra**2 + dec**2)
        
        return significance, true_ang_sep

def main():
    # Set device during generation
    torch.set_default_device('cuda:0')
    torch.set_default_dtype(torch.float64)
    #torch.multiprocessing.set_start_method('spawn')

    loader = PpopLoader('MeaYinPlanetPopulation_Full.txt')

    observer = PoissonObservation(loader)
    observer.evaluate_all()


    #significance = observer.evaluate_stellar_system(stellar_info)
    #print(significance)

if __name__ == '__main__':
    main()