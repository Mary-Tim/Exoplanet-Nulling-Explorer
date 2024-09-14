import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from nullingexplorer.model.spectrum import BaseSpectrum
from nullingexplorer.model.instrument import MiYinBasicType

from spectres import spectres

from tensordict import TensorDict

class BaseInterpolation(ABC, BaseSpectrum):
    def __init__(self, wl_min=5, wl_max=17, num_points=11, interp_points=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('wl_min', torch.tensor(wl_min*1e-6))
        self.register_buffer('wl_max', torch.tensor(wl_max*1e-6))
        self.register_buffer('num_points', torch.tensor(num_points))
        self.register_buffer('equal_width', torch.tensor(True))

        if self.num_points < 2:
            raise ValueError('Number of interpolation points must be greater than 1.')

        if interp_points is not None:
            self.num_points.data.fill_(len(interp_points))
            self.register_buffer('interp_points', torch.tensor(interp_points).contiguous())
            self.equal_width.data.fill_(False)
        else:
            wl_d = (self.wl_max - self.wl_min) / (self.num_points - 1)
            interp_points = [self.wl_min + i * wl_d for i in range(self.num_points)]
            self.register_buffer('interp_points', torch.tensor(interp_points).contiguous())
        # Register flux as a parameter
        self.flux = nn.Parameter(torch.ones(self.num_points))
        self.boundary = {'flux': [torch.zeros(self.num_points), torch.ones(self.num_points)*100.]}

    def get_bin_index(self, wl_mid):
        if self.equal_width:
            wl_min = self.interp_points[0].to(wl_mid.dtype)
            wl_max = self.interp_points[-1].to(wl_mid.dtype)
            delta_width = (wl_max - wl_min) / (self.num_points - 1)
            bins = torch.linspace(wl_min - delta_width, wl_max + delta_width, self.num_points + 2)
            bin_idx = torch.bucketize(wl_mid, bins, right=True)
        else:
            bin_idx = torch.bucketize(wl_mid, self.interp_points, right=True)
        
        bin_idx = bin_idx - 1
        bin_idx = bin_idx.detach()
        return bin_idx

    @abstractmethod
    def load_spectrum(self, data:TensorDict):
        pass

    def forward(self, data:TensorDict):
        return 1
    
class LinearInterpolation(BaseInterpolation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_spectrum(self, input_edge: np.ndarray, input_flux: np.ndarray):
        new_spec = spectres(new_wavs=self.interp_points.cpu().detach().numpy(),
                        spec_wavs=input_edge,
                        spec_fluxes=input_flux,
                    )
        self.flux.data = torch.tensor(new_spec)
        
    @staticmethod
    def add_f(data:TensorDict, xl, xr, yl, yr):
        left = torch.max(data['wl_lo'], xl)
        right = torch.min(data['wl_hi'], xr)
        
        mask = left < right
        
        y_left = yl + (left - xl) / (xr - xl) * (yr - yl)
        y_right = yl + (right - xl) / (xr - xl) * (yr - yl)
        
        flux = torch.where(
            mask,
            0.5 * (y_left + y_right),
            torch.zeros_like(data['wl_mid'])
        )
        
        return flux

    def forward(self, data: TensorDict):
        total_flux = torch.zeros_like(data['wl_mid'])
        total_weight = torch.zeros_like(data['wl_mid'])
        
        # 处理左侧外推
        left_mask = data['wl_hi'] <= self.interp_points[0]
        left_slope = (self.flux[1] - self.flux[0]) / (self.interp_points[1] - self.interp_points[0])
        left_flux = self.flux[0] + left_slope * (data['wl_mid'] - self.interp_points[0])
        total_flux += left_mask * left_flux
        total_weight += left_mask
        
        # 处理右侧外推
        right_mask = data['wl_lo'] >= self.interp_points[-1]
        right_slope = (self.flux[-1] - self.flux[-2]) / (self.interp_points[-1] - self.interp_points[-2])
        right_flux = self.flux[-1] + right_slope * (data['wl_mid'] - self.interp_points[-1])
        total_flux += right_mask * right_flux
        total_weight += right_mask
        
        # 处理插值范围内的部分
        for i in range(self.num_points - 1):
            xl, xr = self.interp_points[i], self.interp_points[i + 1]
            yl, yr = self.flux[i], self.flux[i + 1]
            contribution = torch.vmap(self.add_f, in_dims=(0, None, None, None, None))(data, xl, xr, yl, yr)
            
            # 计算每个区间的权重（区间宽度）
            weight = torch.clamp(torch.min(data['wl_hi'], xr) - torch.max(data['wl_lo'], xl), min=0)
            
            total_flux += contribution * weight
            total_weight += weight
        
        # 计算加权平均flux
        result = torch.where(total_weight > 0, total_flux / total_weight, torch.zeros_like(total_flux))
        return result

#class CubicSplineInterpolation(BaseInterpolation):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
#        if self.num_points < 4:
#            raise ValueError('Number of interpolation points must be at least 4 for cubic spline.')
#
#        # 初始化缓存
#        self.cached_coeffs = None
#        self.cached_flux = None
#
#    def load_spectrum(self, input_edge: np.ndarray, input_flux: np.ndarray):
#        new_spec = spectres(new_wavs=self.interp_points.cpu().detach().numpy(),
#                        spec_wavs=input_edge,
#                        spec_fluxes=input_flux,
#                    )
#        self.flux.data = torch.tensor(new_spec)
#
#        # 重置缓存
#        self.cached_coeffs = None
#        self.cached_flux = None
#
#    def _compute_spline_coefficients(self):
#        # 检查缓存是否有效
#        if self.cached_coeffs is not None and torch.all(self.cached_flux == self.flux):
#            return self.cached_coeffs
#        
#        x = self.interp_points
#        y = self.flux
#        
#        h = x[1:] - x[:-1]
#        a = y[:-1]
#        b = torch.zeros_like(y)
#        d = torch.zeros_like(y)
#        
#        # 构建三对角矩阵
#        A = torch.zeros((self.num_points, self.num_points), device=x.device)
#        
#        # 实现 not-a-knot 条件
#        A[0, 0:3] = torch.tensor([h[1], -(h[0] + h[1]), h[0]])
#        A[-1, -3:] = torch.tensor([h[-2], -(h[-2] + h[-1]), h[-1]])
#        
#        for i in range(1, self.num_points - 1):
#            A[i, i-1:i+2] = torch.tensor([h[i-1], 2*(h[i-1]+h[i]), h[i]])
#        
#        # 构建右侧向量
#        r = torch.zeros_like(y)
#        r[0] = ((y[1] - y[0]) / h[0] - (y[2] - y[1]) / h[1]) * h[1]
#        r[-1] = ((y[-2] - y[-3]) / h[-2] - (y[-1] - y[-2]) / h[-1]) * h[-2]
#        for i in range(1, self.num_points - 1):
#            r[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
#        
#        
#        # 解三对角矩阵方程
#        c = torch.linalg.solve(A, r)
#        
#        for i in range(self.num_points - 1):
#            b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
#            d[i] = (c[i+1] - c[i]) / (3 * h[i])
#
#        # 更新缓存
#        self.cached_coeffs = (a, b, c, d)
#        self.cached_flux = self.flux.clone()
#        
#        return a, b, c, d
#
#    def _interpolate(self, x_new):
#        a, b, c, d = self._compute_spline_coefficients()
#        
#        # 找到x_new落在哪个区间
#        idx = torch.searchsorted(self.interp_points, x_new) - 1
#        idx = torch.clamp(idx, 0, len(self.interp_points) - 2)
#        
#        # 计算相对x位置
#        x_rel = x_new - self.interp_points[idx]
#        
#        # 计算插值结果
#        y_new = a[idx] + b[idx] * x_rel + c[idx] * x_rel**2 + d[idx] * x_rel**3
#        
#        return y_new
#
#    def forward(self, data: TensorDict):
#        return self._interpolate(data['wl_mid'])
#
#    def integral(self, wl_min, wl_max):
#        a, b, c, d = self._compute_spline_coefficients()
#        
#        def primitive(x, idx):
#            x_rel = x - self.interp_points[idx]
#            return a[idx] * x_rel + b[idx] * x_rel**2 / 2 + c[idx] * x_rel**3 / 3 + d[idx] * x_rel**4 / 4
#        
#        idx_min = torch.searchsorted(self.interp_points, wl_min) - 1
#        idx_max = torch.searchsorted(self.interp_points, wl_max) - 1
#        idx_min = torch.clamp(idx_min, 0, len(self.interp_points) - 2)
#        idx_max = torch.clamp(idx_max, 0, len(self.interp_points) - 2)
#        
#        integral = torch.zeros_like(wl_min)
#        
#        for i in range(self.num_points - 1):
#            x_left = torch.max(wl_min, self.interp_points[i])
#            x_right = torch.min(wl_max, self.interp_points[i+1])
#            mask = x_right > x_left
#            integral += mask * (primitive(x_right, i) - primitive(x_left, i))
#        
#        return integral


class CubicSplineInterpolation(BaseInterpolation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_points < 4:
            raise ValueError('Number of interpolation points must be at least 4 for cubic spline.')

        # 初始化缓存
        self.cached_coeffs = None
        self.cached_flux = None

    def load_spectrum(self, input_edge: np.ndarray, input_flux: np.ndarray):
        new_spec = spectres(new_wavs=self.interp_points.cpu().detach().numpy(),
                        spec_wavs=input_edge,
                        spec_fluxes=input_flux,
                    )
        self.flux.data = torch.tensor(new_spec)

        # 重置缓存
        self.cached_coeffs = None
        self.cached_flux = None

    def _compute_spline_coefficients(self):
        # 检查缓存是否有效
        if self.cached_coeffs is not None and torch.all(self.cached_flux == self.flux):
            return self.cached_coeffs
        
        x = self.interp_points
        y = self.flux
        
        h = x[1:] - x[:-1]
        a = y[:-1]

        # 构建矩阵 A
        A = torch.zeros((self.num_points, self.num_points), device=x.device)
        
        # not-a-knot 条件
        A[0, 0:3] = torch.tensor([h[1], -(h[0] + h[1]), h[0]])
        A[-1, -3:] = torch.tensor([h[-2], -(h[-2] + h[-1]), h[-1]])
        
        # 中间行
        i = torch.arange(1, self.num_points-1)
        A[i, i-1] = h[:-1]
        A[i, i] = 2 * (h[:-1] + h[1:])
        A[i, i+1] = h[1:]

        # 构建右侧向量 r
        r = torch.zeros_like(y)
        r[1:-1] = 3 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        r[0] = ((y[1] - y[0]) / h[0] - (y[2] - y[1]) / h[1]) * h[1]
        r[-1] = ((y[-2] - y[-3]) / h[-2] - (y[-1] - y[-2]) / h[-1]) * h[-2]

        # 解三对角矩阵方程
        c = torch.linalg.solve(A, r)

        b = (y[1:] - y[:-1]) / h - h * (2*c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3 * h)

        self.cached_coeffs = (a, b, c[:-1], d)
        self.cached_flux = self.flux.clone()

        return self.cached_coeffs

    def _interpolate(self, x_new):
        a, b, c, d = self._compute_spline_coefficients()
        
        idx = torch.searchsorted(self.interp_points, x_new) - 1
        idx = torch.clamp(idx, 0, len(self.interp_points) - 2)
        
        x_rel = x_new - self.interp_points[idx]
        
        y_new = a[idx] + b[idx] * x_rel + c[idx] * x_rel**2 + d[idx] * x_rel**3
        
        return y_new

    def forward(self, data: TensorDict):
        return self._interpolate(data['wl_mid'])