import torch
import torch.nn as nn

from nullingexplorer.model.spectrum import BaseSpectrum
from nullingexplorer.model.instrument import MiYinBasicType
from nullingexplorer.io import MiYinData

class BaseInterpolation(BaseSpectrum):
    def __init__(self, wl_min=7.e-6, wl_max=20.e-6, num_points=10, points=None):
        super().__init__()

        self.wl_min = wl_min
        self.wl_max = wl_max
        self.num_points = num_points

        if points is None:
            wl_d = (self.wl_max - self.wl_min) / (self.num_points - 1)
            points = [self.wl_min + i * wl_d for i in range(self.num_points)]
        else:
            self.wl_min = torch.min(points)
            self.wl_max = torch.max(points)
            self.num_points = len(points)

        if self.num_points < 2:
            raise ValueError('Number of interpolation points must be greater than 2.')

        if isinstance(points) is torch.Tensor:
            self.register_buffer('interp_x', points)
        else:
            self.register_buffer('interp_x', torch.tensor(points))
        self.register_parameter('interp_y', nn.Parameter(torch.ones(self.num_points)))


    def init_values(self, data: MiYinData):
        # Initialize the value of interpolation points.
        instrument = MiYinBasicType()
        wl_center = data.get_bins('wavelength')
        wl_bounds = wl_center - data.get_bins('wl_width') / 2.
        buffer = torch.bucketize(self.interp_x.item(), wl_bounds)
        buffer[buffer < 0] = 0
        inty_val = []
        for bf in buffer:
            wl_data = data.select_data('wavelength', wl_center[bf])
            inty_val.append(torch.mean(wl_data.photon_electron/instrument.forward(wl_data)))
        setattr(self, 'interp_y', nn.Parameter(inty_val))

    def forward():
        return 1


class CubicSplineInterpolation(BaseInterpolation):
    def __init__(self, wl_min=7e-6, wl_max=20e-6, num_points=10, points=None):
        super().__init__(wl_min, wl_max, num_points, points)




