import os
import h5py
import numpy as np
import mplhep
import yaml
import iminuit

from datetime import datetime
from scipy import optimize, stats
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use(mplhep.style.LHCb2)

#from nullingexplorer.fitter import NegativeLogLikelihood
from nullingexplorer.utils import Constants as cons

class FitResult():
    def __init__(self, auto_save=True, output_path = None, *args, **kwargs):
        self.__result = {}
        self.__unit = {
                        'ra': 'mas',
                        'dec': 'mas',
                        'radius': 'kilometer',
                        'temperature': 'Kelvin',
                        'r_ra': '100 * mas',
                        'r_dec': '100 * mas',
                        'r_radius': '6371e3 * kilometer',
                        'r_temperature': '285 * Kelvin',
                        'r_angular': '100 * mas',
                        'r_polar': 'radian',
                        'au': 'AU',
                        'polar': 'radian',
                       }
        self.nll_model = None
        self.__auto_save = auto_save
        if self.__auto_save:
            start_time = datetime.now()
            if output_path is None:
                self.__output_path = f"results/Job_{start_time.strftime('%Y%m%d_%H%M%S')}"
            else:
                self.__output_path = f"{output_path}/Job_{start_time.strftime('%Y%m%d_%H%M%S')}"
            if not os.path.exists(self.__output_path):
                os.makedirs(self.__output_path)
            print(f"The result will be saved to: {self.__output_path}")

    def load_fit_result(self, nll_model, scipy_result: optimize.OptimizeResult):
        if hasattr(scipy_result, "lowest_optimization_result"):
            scipy_result = scipy_result.lowest_optimization_result
        self.nll_model = nll_model

        # The final NLL
        self.__result['best_nll'] = scipy_result.fun
        # The name of parameters
        self.__result['param_name'] = [name[name.find(".")+1:] for name in self.nll_model.name_of_params_flat]
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

    def load_minuit_result(self, nll_model, minuit_result: iminuit.Minuit):
    #def save_fit_result(self, nll_model: NegativeLogLikelihood, minuit_result: optimize.OptimizeResult):
        self.nll_model = nll_model

        # Set model parameters to the final values
        for name, val in zip(self.nll_model.free_param_list.keys(), minuit_result.values):
            self.nll_model.set_param_val(name, val)
        # The final NLL
        self.__result['best_nll'] = minuit_result.fval
        # The name of parameters
        #self.__result['param_name'] = self.nll_model.free_param_list.keys()
        self.__result['param_name'] = [name[name.find(".")+1:] for name in self.nll_model.free_param_list.keys()]
        # The fit result of parameters
        self.__result['param_val'] = np.array(minuit_result.values)
        # set units
        self.__result['param_unit'] = []
        for name in self.__result['param_name']:
            param_end = name.split('.')[-1]
            if param_end in self.__unit.keys():
                self.__result['param_unit'].append(self.__unit[param_end])
            else:
                self.__result['param_unit'].append('')

    def load_mcmc_result(self, nll_model, samples, results:dict):
        self.nll_model = nll_model
    
        # 保存参数名称和单位
        self.__result['param_name'] = list(samples.keys())
        self.__result['param_unit'] = []
        for name in self.__result['param_name']:
            param_end = name.split('.')[-1]
            if param_end in self.__unit.keys():
                self.__result['param_unit'].append(self.__unit[param_end])
            else:
                self.__result['param_unit'].append('')
    
        # 计算最佳拟合值及误差
        self.__result['param_val'] = np.zeros(len(self.__result['param_name']))
        self.__result['std_err'] = np.zeros(len(self.__result['param_name']))
        for i, param in enumerate(self.__result['param_name']):
            self.__result['param_val'][i] = np.mean(samples[param])
            self.__result['std_err'][i] = np.std(samples[param])
            self.__result[f'{param}_samples'] = samples[param]
    
        # 保存完整的采样链以供后续分析
        #self.__result['mcmc_samples'] = samples

    def evaluate_std_error(self):
        # Standard uncertainties
        std_err, inv_hess = self.nll_model.std_error()
        self.__result['std_err'] = std_err.cpu().detach().numpy()
        self.__result['inverse_hessian'] = inv_hess.cpu().detach().numpy()
        for i, name in enumerate(self.__result['param_name']):
            if name.endswith(('.ra', '.dec')):
                self.__result['param_val'][i] *= cons._radian_to_mas
                self.__result['std_err'][i] *= cons._radian_to_mas
        return self.__result['std_err']

    def set_item(self, key, val):
        self.__result[key] = val

    def get_item(self, key):
        if key not in self.__result.keys():
            raise KeyError(f'Item {key} not exist.')

        return self.__result[key]

    def save(self, name=None, save_type = 'hdf5'):
        if self.__auto_save == False:
            return
        save_name = f"result"
        if name is not None:
            save_name = f"{save_name}_{name}"
        if save_type == 'hdf5':
            self.__auto_save_hdf5(save_name)
        elif save_type == 'fits':
            self.__auto_save_fits(save_name)
        else:
            raise TypeError('File format not support')
        print(f"Save fit result to: {self.__output_path}/{save_name}.{save_type}")


    def __auto_save_hdf5(self, name):
        with h5py.File(f"{self.__output_path}/{name}.hdf5", 'w') as file:
            for key, val in self.__result.items():
                try:    
                    file.create_dataset(key, data=val)
                except:
                    print(f"Warning! Cannot save {key} to hdf5 file. Data type: {type(val)}")

    def __auto_save_fits(self, name):
        '''
        TODO: 类比__auto_save_hdf5，将self.__result存入fits文件
        '''
        pass

    def keys(self):
        return self.__result.keys()

    @property
    def result(self):
        return self.__result

    @classmethod
    def load(cls, path: str):
        result = cls(auto_save=False)
        with h5py.File(f"{path}", 'r') as file:
            dataset_list = file.keys()
            for ds in dataset_list:
                result.set_item(ds, file[ds][()])

        if 'param_name' in result.keys():
            param_name = [name.decode("utf-8") for name in result.get_item('param_name')]
            result.set_item('param_name', param_name)

        if 'param_unit' in result.keys():
            param_unit = [unit.decode("utf-8") for unit in result.get_item('param_unit')]
            result.set_item('param_unit', param_unit)

        return result

    def print_result(self):
        if 'best_nll' in self.__result.keys():
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

    def draw_scan_result(self, position_name=[], file_name='fitter_ramdon', show=False, polar=False, *args, **kwargs):
        if 'scan_nll' not in self.__result.keys():
            raise KeyError('Scan result not found.')

        fig, ax = plt.subplots()
        line = np.arange(0, len(self.__result['scan_nll']))
        scat = ax.scatter(line, self.__result['scan_nll'], s=30)
        ax.set_xlabel("Task")
        ax.set_ylabel("NLL")
        if self.__auto_save:
            plt.savefig(f'{self.__output_path}/{file_name}_NLL.pdf')

        if len(position_name) != 0:
            ra_index  = self.__result['param_name'].index(position_name[0])
            dec_index = self.__result['param_name'].index(position_name[1])
            if ra_index == -1 or dec_index == -1:
                raise KeyError(f'Position name {position_name[0]} and/or {position_name[1]} not found.')
            ra_array = self.__result['scan_result'][:,ra_index]
            dec_array = self.__result['scan_result'][:,dec_index]

            seq_index = np.argsort(-self.__result['scan_nll'])
            ra_array = ra_array[seq_index]
            dec_array = dec_array[seq_index]
            nll_array = self.__result['scan_nll'].copy()
            nll_array = nll_array[seq_index]
            fig = plt.figure()
            ax = plt.subplot(111, polar=polar)
            #levels = np.arange(np.min(nll_array)*1.005, 10., np.fabs(np.max(nll_array)-np.min(nll_array))/100.)
            colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 0.75, 256))
            my_cmap = LinearSegmentedColormap.from_list('new_cmap', colors)
            scat = ax.scatter(ra_array, dec_array, s=30, c=nll_array, cmap=my_cmap)
            fig.colorbar(scat,ax=ax,orientation='vertical',label='NLL')
            ax.plot(ra_array[-1], dec_array[-1], 'r*', markersize=15, label='Min NLL')
            ax.legend(fontsize='xx-large', loc='lower right', bbox_to_anchor=(1.1, -0.05))
            if not polar:
                ax.set_xlabel(position_name[0])
                ax.set_ylabel(position_name[1])
            if self.__auto_save:
                plt.savefig(f'{self.__output_path}/{file_name}_location.pdf')

            if show:
                plt.show()

    def dump_config(self, config_name, config: dict):
        with open(f"{self.__output_path}/{config_name}.yaml", 'w') as file:
            yaml.dump(config, file)

    def draw_meshgrid_result(self, x_grid, y_grid, nll_grid):
        fig, ax = plt.subplots()
        levels = np.arange(np.min(nll_grid)*1.005, 10., np.fabs(np.max(nll_grid)-np.min(nll_grid))/100.)
        trans_map_cont = ax.contourf(x_grid, y_grid, nll_grid, levels=levels, cmap = plt.get_cmap("bwr"))
        ax.set_xlabel("ra / mas")
        ax.set_ylabel("dec / mas")
        
        fig.colorbar(trans_map_cont)
        
        if self.__auto_save:
            plt.savefig(f'{self.__output_path}/scan_nll.pdf')
        plt.show()

    def plot_corner(self, samples, params=None, show=True):
        """绘制MCMC采样结果的corner plot

        Args:
            samples: MCMC采样结果字典
            params: 需要绘制的参数列表,如果为None则绘制所有参数
            save_path: 保存路径,如果为None则不保存
            show: 是否显示图像
        """
        try:
            import corner
        except ImportError:
            print("请先安装corner包: pip install corner")
            return

        plt.style.use('default')

        # 如果没有指定samples,则使用self.__result中的采样结果
        if samples is None:
            samples = {}
            for name in self.__result['param_name']:
                samples[name] = self.__result[f'{name}_samples']
        # 如果没有指定参数,则使用所有参数
        if params is None:
            params = list(samples.keys())
            # 移除'obs'键(如果存在)
            if 'obs' in params:
                params.remove('obs')


        # 准备数据
        data = np.vstack([samples[param] for param in params]).T

        # 准备标签
        labels = []
        for param in params:
            # 获取参数单位(如果有)
            unit = ""
            if param.endswith(('.ra', '.dec')):
                unit = " [mas]"
            elif param.endswith('.au'):
                unit = " [AU]"
            elif param.endswith('.radius'):
                unit = " [km]"
            elif param.endswith('.temperature'):
                unit = " [K]"
            tail = param.split('.')[-1]
            labels.append(f"{tail}{unit}")

        # 创建figure
        fig = corner.corner(
            data,
            labels=labels,
            show_titles=True,  # 显示1D分布的统计信息
            title_fmt='.2f',   # 统计值的精度
            title_kwargs={"fontsize": 12},
            #label_kwargs={"fontsize": 12},
            quantiles=[0.16, 0.5, 0.84],  # 显示16%,50%,84%分位数
            #levels=[0.68, 0.95],          # 等高线的置信区间
            #plot_datapoints=True,         # 绘制采样点
            #plot_density=True,            # 绘制密度估计
            #fill_contours=True,           # 填充等高线
            #use_math_text=True,           # 使用LaTeX格式
        )

        # 调整图像大小
        #fig.set_size_inches(12, 12)

        # 保存图像
        if self.__auto_save:
            fig.savefig(f'{self.__output_path}/corner.pdf', bbox_inches='tight', dpi=300)

        # 显示图像
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig