from scipy import optimize, stats
import h5py

from nullingexplorer.fitter import NegativeLogLikelihood
from nullingexplorer.utils import Constants as cons

class FitResult():
    def __init__(self):
        self.__result = {}
        self.__unit = {'ra': 'mas',
                       'dec': 'mas',
                       'radius': 'kilometer',
                       'temperature': 'Kelvin'
                       }

    def save_fit_result(self, nll_model: NegativeLogLikelihood, scipy_result: optimize.OptimizeResult):
        if hasattr(scipy_result, "lowest_optimization_result"):
            scipy_result = scipy_result.lowest_optimization_result

        # Set model parameters to the final values
        for name, val in zip(nll_model.free_param_list.keys(), scipy_result.x):
            nll_model.set_param_val(name, val)
        # The final NLL
        self.__result['best_nll'] = scipy_result.fun
        # The name of parameters
        self.__result['param_name'] = [name[name.find(".")+1:] for name in nll_model.free_param_list.keys()]
        # The fit result of parameters
        self.__result['param_val'] = scipy_result.x
        # Standard uncertainties
        self.__result['std_err'] = nll_model.std_error().cpu().detach().numpy()
        for i, name in enumerate(self.__result['param_name']):
            if name.endswith(('.ra', '.dec')):
                self.__result['param_val'][i] *= cons._radian_to_mas
                self.__result['std_err'][i] *= cons._radian_to_mas
        # inverse hessian matrix
        self.__result['inverse_hessian'] = nll_model.inverse_hessian().cpu().detach().numpy()
        # set units
        self.__result['param_unit'] = []
        for name in self.__result['param_name']:
            param_end = name.split('.')[-1]
            if param_end in self.__unit.keys():
                self.__result['param_unit'].append(self.__unit[param_end])
            else:
                self.__result['param_unit'].append('')

    def set_item(self, key, val):
        self.__result[key] = val

    def get_item(self, key):
        if key not in self.__result.keys():
            raise KeyError(f'Item {key} not exist.')

        return self.__result[key]

    def save(self, path: str):
        if path.endswith(('.hdf5', '.h5',)):
            self.__save_hdf5(path)
        elif path.endswith('.fits'):
            self.__save_fits(path)
        else:
            raise TypeError('File format not support')
        print(f"Save fit result to: {path}.")


    def __save_hdf5(self, path: str):
        with h5py.File(path, 'w') as file:
            for key, val in self.__result.items():
                file.create_dataset(key, data=val)

    def __save_fits(self, path: str):
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
        print(f"Inverse hessian matrix:\n{self.__result['inverse_hessian']}")
        for name, val, err, unit in zip(self.__result['param_name'], self.__result['param_val'], self.__result['std_err'], self.__result['param_unit']):
            print(f"{name}:\t{val:.3f} +/- {err:.3f}\t[{unit}]")

    def significance(self, bkg=-1000, ndf=None, sig=None):
        if sig == None:
            sig = self.__result['best_nll']
        if ndf == None:
            ndf = len(self.__result['param_val'])

        delta_2ll = 2 * abs(sig-bkg)
        n_sigma = -stats.norm.ppf(stats.chi2.sf(delta_2ll,df=ndf,loc=0,scale=1)/2)
        return n_sigma