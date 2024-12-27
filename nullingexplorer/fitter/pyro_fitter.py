import pyro
import pyro.distributions as dist
import torch
import numpy as np  

from tensordict import TensorDict
from nullingexplorer.model.amplitude import BaseAmplitude
from nullingexplorer.fitter import GaussianNLL, PoissonNLL
from nullingexplorer.io import FitResult

class PyroFitter:
    NLL_dict = {
        'gaussian': GaussianNLL,
        'poisson': PoissonNLL
    }
    
    def __init__(self, amp: BaseAmplitude, data: TensorDict, NLL_type='gaussian'):
        self.amp = amp
        self.data = data.detach()
        if NLL_type not in self.NLL_dict.keys():
            raise KeyError(f"NLL type {NLL_type} not found!")
        self.NLL = self.NLL_dict[NLL_type](self.amp, self.data)
        self.fit_result = FitResult()

    def model(self):
        # 获取参数边界
        boundaries = self.NLL.get_boundaries()
        params = {}
    
        # 为每个参数创建先验分布
        for name, (lower, upper) in zip(self.NLL.name_of_params, boundaries):
            # 使用均匀分布作为先验
            params[name] = pyro.sample(
                name,
                dist.Uniform(torch.tensor(lower), torch.tensor(upper))
            )
    
        # 更新模型参数
        self.NLL.update_vals(list(params.values()))
    
        # 计算似然
        #predicted = self.NLL.call_nll()
        predicted = self.NLL.amp(self.data)
    
        # 根据不同的似然函数类型定义观测
        if isinstance(self.NLL, GaussianNLL):
            sample_model = pyro.sample(
                "obs",
                dist.Normal(predicted, self.data['pe_uncertainty']),
                obs=self.data['photon_electron']
            )
        elif isinstance(self.NLL, PoissonNLL):
            sample_model = pyro.sample(
                "obs",
                dist.Poisson(predicted),
                obs=self.data['photon_electron']
            )

        return sample_model

    def run_mcmc(self, num_samples=1000, warmup_steps=200):
        """运行 MCMC 采样

        Args:
            num_samples: 采样数量
            warmup_steps: 预热步数
        """
        # 设置 NUTS 核采样器
        kernel = pyro.infer.NUTS(self.model)
        mcmc = pyro.infer.MCMC(
            kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps
        )
    
        # 运行采样
        mcmc.run()
    
        # 获取采样结果
        samples = {k: v.detach().cpu().numpy() 
                  for k, v in mcmc.get_samples().items()}
    
        return samples

    def search_planet(self, amp_name: str, num_samples=1000, warmup_steps=200,
                 draw=False, show=False):
        print(f"搜索行星 {amp_name}...")
    
        # 获取相关参数
        self.NLL.free_all_params()
        name_of_params = self.NLL.name_of_params
        planet_params = []
        for name in name_of_params:
            if name.find(amp_name) != -1:
                planet_params.append(name)
        if len(planet_params) == 0:
            raise KeyError(f"Planet {amp_name} not found!")

        self.NLL.config_fit_params(planet_params)
    
        # 运行 MCMC
        samples = self.run_mcmc(num_samples, warmup_steps)
    
        # 计算参数的后验均值和标准差
        results = {}
        for param in planet_params:
            mean = np.mean(samples[param])
            std = np.std(samples[param])
            results[param] = {'mean': mean, 'std': std}

            # 更新最佳拟合值
            self.NLL.set_param_val(param, mean)
    
        # 保存结果
        self.fit_result.load_mcmc_result(self.NLL, samples, results)
        self.fit_result.print_result()
        self.fit_result.save(name=amp_name)
    
        # 绘制结果
        if draw:
            self.fit_result.plot_corner(samples, show=show)
    
        return results