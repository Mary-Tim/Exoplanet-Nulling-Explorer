from scipy import optimize, stats
import h5py

from nullingexplorer import NegativeLogLikelihood

class FitResult():
    def __init__(self):
        self.__result = {}

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
        # inverse hessian matrix
        self.__result['inverse_hessian'] = nll_model.inverse_hessian().cpu().detach().numpy()

    def set_item(self, key, val):
        self.__result[key] = val

    def get_item(self, key):
        if key not in self.__result.keys():
            raise KeyError(f'Item {key} not exist.')

        return self.__result[key]

    def save(self, path: str, file_name='results.hdf5'):
        if not path.endswith(('.hdf5', '.h5',)):
            if path.endswith('/'):
                path = f"{path}{file_name}"
            else:
                path = f"{path}/{file_name}"

        with h5py.File(path, 'w') as file:
            for key, val in self.__result.items():
                file.create_dataset(key, data=val)
            file.close()

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

        return result

    def print_result(self):
        print(f"LL: {self.__result['best_nll']:.03f}")
        print(f"Inverse hessian matrix:\n{self.__result['inverse_hessian']}")
        for name, val, err in zip(self.__result['param_name'], self.__result['param_val'], self.__result['std_err']):
            print(f"{name}:\t{val:.3f} +/- {err:.3f}")

    @classmethod
    def significance(cls, ndf=2, sig=-1000, bkg=-1000):

        delta_2ll = 2 * abs(sig-bkg)
        n_sigma = -stats.norm.ppf(stats.chi2.sf(delta_2ll,df=ndf,loc=0,scale=1)/2)
        return n_sigma