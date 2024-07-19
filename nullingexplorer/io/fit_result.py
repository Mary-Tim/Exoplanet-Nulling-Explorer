import os
import h5py
import numpy as np
import mplhep

from datetime import datetime
from scipy import optimize, stats
from matplotlib import pyplot as plt
plt.style.use(mplhep.style.LHCb2)

#from nullingexplorer.fitter import NegativeLogLikelihood
from nullingexplorer.utils import Constants as cons

class FitResult():
    def __init__(self, output_path = None):
        self.__result = {}
        self.__unit = {'ra': 'mas',
                       'dec': 'mas',
                       'radius': 'kilometer',
                       'temperature': 'Kelvin'
                       }
        self.nll_model = None
        start_time = datetime.now()
        if output_path is None:
            self.__output_path = f"results/Job_{start_time.strftime('%Y%m%d_%H%M%S')}"
        else:
            self.__output_path = f"{output_path}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(self.__output_path):
            os.mkdir(self.__output_path)

    def load_fit_result(self, nll_model, scipy_result: optimize.OptimizeResult):
    #def save_fit_result(self, nll_model: NegativeLogLikelihood, scipy_result: optimize.OptimizeResult):
        if hasattr(scipy_result, "lowest_optimization_result"):
            scipy_result = scipy_result.lowest_optimization_result
        self.nll_model = nll_model

        # Set model parameters to the final values
        for name, val in zip(self.nll_model.free_param_list.keys(), scipy_result.x):
            self.nll_model.set_param_val(name, val)
        # The final NLL
        self.__result['best_nll'] = scipy_result.fun
        # The name of parameters
        #self.__result['param_name'] = self.nll_model.free_param_list.keys()
        self.__result['param_name'] = [name[name.find(".")+1:] for name in self.nll_model.free_param_list.keys()]
        # The fit result of parameters
        self.__result['param_val'] = scipy_result.x
        # set units
        self.__result['param_unit'] = []
        for name in self.__result['param_name']:
            param_end = name.split('.')[-1]
            if param_end in self.__unit.keys():
                self.__result['param_unit'].append(self.__unit[param_end])
            else:
                self.__result['param_unit'].append('')

    def evaluate_std_error(self):
        # Standard uncertainties
        self.__result['std_err'] = self.nll_model.std_error().cpu().detach().numpy()
        for i, name in enumerate(self.__result['param_name']):
            if name.endswith(('.ra', '.dec')):
                self.__result['param_val'][i] *= cons._radian_to_mas
                self.__result['std_err'][i] *= cons._radian_to_mas
        # inverse hessian matrix
        self.__result['inverse_hessian'] = self.nll_model.inverse_hessian().cpu().detach().numpy()
        return self.__result['std_err']

    def set_item(self, key, val):
        self.__result[key] = val

    def get_item(self, key):
        if key not in self.__result.keys():
            raise KeyError(f'Item {key} not exist.')

        return self.__result[key]

    def save(self, save_type = 'hdf5'):
        if save_type == 'hdf5':
            self.__save_hdf5()
        elif save_type == 'fits':
            self.__save_fits()
        else:
            raise TypeError('File format not support')
        print(f"Save fit result to: {self.__output_path}/result.{save_type}")


    def __save_hdf5(self):
        with h5py.File(f"{self.__output_path}/result.hdf5", 'w') as file:
            for key, val in self.__result.items():
                file.create_dataset(key, data=val)

    def __save_fits(self):
        '''
        TODO: 类比__save_hdf5，将self.__result存入fits文件
        '''
        pass


    def keys(self):
        return self.__result.keys()

    @classmethod
    def load(cls, path: str):
        result = cls()
        with h5py.File(f"{path}", 'r') as file:
            dataset_list = file.keys()
            for ds in dataset_list:
                result.set_item(ds, file[ds][:])
            file.close()

        if 'param_name' in result.keys():
            param_name = [name.decode("utf-8") for name in result.get_item('param_name')]
            result.set_item('param_name', param_name)

        if 'param_unit' in result.keys():
            param_unit = [unit.decode("utf-8") for unit in result.get_item('param_unit')]
            result.set_item('param_unit', param_unit)

        return result

    def print_result(self):
        print(f"LL: {self.__result['best_nll']:.03f}")
        if 'inverse_hessian' in self.__result.keys():
            print(f"Inverse hessian matrix:\n{self.__result['inverse_hessian']}")
        if 'std_err' in self.__result.keys():
            for name, val, err, unit in zip(self.__result['param_name'], self.__result['param_val'], self.__result['std_err'], self.__result['param_unit']):
                print(f"{name}:\t{val:.3f} +/- {err:.3f}\t[{unit}]")
        else:
            for name, val, unit in zip(self.__result['param_name'], self.__result['param_val'], self.__result['param_unit']):
                print(f"{name}:\t{val:.3f}\t[{unit}]")


    def significance(self, bkg=-1000, ndf=None, sig=None):
        if sig == None:
            sig = self.__result['best_nll']
        if ndf == None:
            ndf = len(self.__result['param_val'])

        delta_2ll = 2 * abs(sig-bkg)
        n_sigma = -stats.norm.ppf(stats.chi2.sf(delta_2ll,df=ndf,loc=0,scale=1)/2)
        return n_sigma

    def draw_scan_result(self, position_name=[]):
        if 'scan_nll' not in self.__result.keys():
            raise KeyError('Scan result not found.')

        fig, ax = plt.subplots()
        line = np.arange(0, len(self.__result['scan_nll']))
        scat = ax.scatter(line, self.__result['scan_nll'], s=30)
        ax.set_xlabel("Task")
        ax.set_ylabel("NLL")
        plt.savefig(f'{self.__output_path}/NLL.pdf')

        if len(position_name) != 0:
            ra_index  = self.__result['param_name'].index(position_name[0][position_name[0].find(".")+1:])
            dec_index = self.__result['param_name'].index(position_name[1][position_name[1].find(".")+1:])
            #dec_index = self.__result['param_name'].index(position_name[1])
            if ra_index == -1 or dec_index == -1:
                raise KeyError(f'Position name {position_name[0]} and/or {position_name[1]} not found.')
            ra_array = self.__result['scan_result'][:,ra_index]           
            dec_array = self.__result['scan_result'][:,dec_index]

            seq_index = np.argsort(-self.__result['scan_nll'])
            ra_array = ra_array[seq_index]
            dec_array = dec_array[seq_index]
            nll_array = self.__result['scan_nll'].copy()
            nll_array = nll_array[seq_index]
            fig, ax = plt.subplots()
            levels = np.arange(np.min(nll_array)*1.005, 10., np.fabs(np.max(nll_array)-np.min(nll_array))/100.)
            scat = ax.scatter(ra_array, dec_array, s=30, c=nll_array, cmap=plt.get_cmap("bwr"))
            fig.colorbar(scat,ax=ax,orientation='vertical',label='NLL')
            ax.set_xlabel("ra")
            ax.set_ylabel("dec")
            plt.savefig(f'{self.__output_path}/fitter_random.pdf')

            plt.show()


        