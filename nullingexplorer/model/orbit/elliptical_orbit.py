import torch
import torch.nn as nn
from scipy.optimize import newton
import numpy as np

from .base_orbit import BaseOrbit
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.utils import get_transmission

class EllipticalOrbit(BaseOrbit):
    def __init__(self):
        super().__init__()
        # Free parameters
        self.semi_major_axis = nn.Parameter(torch.tensor(1.0))          # Semi-major axis (AU)
        self.eccentricity = nn.Parameter(torch.tensor(0.1))             # Eccentricity
        self.mean_anomaly_at_unix = nn.Parameter(torch.tensor(0.0))     # Mean anomaly at epoch (radians) Epoch: POSIX timestamp start time 1970-1-1 0:0:0 UTC
        self.inclination = nn.Parameter(torch.tensor(0.0))              # Orbital inclination (radians)
        self.longitude_of_ascending_node = nn.Parameter(torch.tensor(0.0))  # Longitude of the ascending node (radians)
        self.argument_of_periastron = nn.Parameter(torch.tensor(0.0))   # Argument of periastron (radians)

        # Boundary of parameters
        self.boundary = {
            'semi_major_axis':          torch.tensor([0.1, 10.0]),
            'eccentricity':             torch.tensor([0.0, 0.99]),
            'mean_anomaly_at_unix':     torch.tensor([0.0, 2 * torch.pi]),  # Mean anomaly at epoch between 0 and 2π
            'inclination':              torch.tensor([0.0, torch.pi]),      # Orbital inclination between 0 and π
            'longitude_of_ascending_node': torch.tensor([0.0, 2 * torch.pi]),  # Longitude of the ascending node between 0 and 2π
            'argument_of_periastron':   torch.tensor([0.0, 2 * torch.pi]),  # Argument of periastron between 0 and 2π
        }

        # Constant parameters
        self.register_buffer('distance', cfg.get_property('distance'))  # Distance between the target and the solar system (pc)
        self.register_buffer('target_longitude', ((cfg.get_property('target_longitude') - 180) % 360 - 180) / cons._radian_to_degree)
        self.register_buffer('target_latitude', cfg.get_property('target_latitude') / cons._radian_to_degree)
        self.register_buffer('au_to_mas', (cons._au_to_meter) / (self.distance) * cons._radian_to_mas)

        # Calculate the host star coordinates
        x_host = self.distance * cons._pc_to_au * torch.cos(self.target_latitude) * torch.cos(self.target_longitude)
        y_host = self.distance * cons._pc_to_au * torch.cos(self.target_latitude) * torch.sin(self.target_longitude)
        z_host = self.distance * cons._pc_to_au * torch.sin(self.target_latitude)
        self.register_buffer('host_pos', torch.tensor([x_host, y_host, z_host]))

        self._au = None

    def forward(self, data):
        # Get unique time values
        unique_times, inverse_indices = torch.unique(data['start_time'], return_inverse=True)
        
        # Calculate the orbital position for each unique time
        eccentric_anomalies = self.calculate_eccentric_anomaly(unique_times)
        self._au = self.semi_major_axis * (1 - self.eccentricity * torch.cos(eccentric_anomalies))
        ra, dec = self.calculate_position(eccentric_anomalies)
        
        # Return the corresponding positions based on the original times
        ra_list = ra[inverse_indices]
        dec_list = dec[inverse_indices]
        
        return ra_list, dec_list

    @property
    def au(self):
        if self._au is None:
            eccentric_anomalies = self.calculate_eccentric_anomaly(self.start_time)
            self._au = self.semi_major_axis * (1 - self.eccentricity * torch.cos(eccentric_anomalies))
        return self._au

    @staticmethod
    def trans_epoch_of_periastron_to_mean_anomaly_at_unix(epoch_of_periastron, semi_major_axis):
        '''
        Transform the epoch of periastron (in Julian Date) to mean anomaly at POSIX timestamp (1970-1-1 0:0:0 UTC).

        Parameters:
        epoch_of_periastron (float): The epoch of periastron in Julian Date (unit: days).
        semi_major_axis (torch.Tensor): The semi-major axis of the orbit (unit: AU).

        Returns:
        torch.Tensor: The mean anomaly at the POSIX timestamp (1970-1-1 0:0:0 UTC) (unit: radians).
        '''
        from astropy.time import Time
        
        # Convert the Julian Date to POSIX timestamp
        epoch_of_periastron_time = Time(epoch_of_periastron, format='jd')
        epoch_of_periastron_unix = epoch_of_periastron_time.unix

        # Calculate the orbital period
        period = np.sqrt(semi_major_axis**3) * cons._year_to_second  # Orbital period (unit: seconds)
        
        # Calculate the mean anomaly
        mean_anomaly = (2 * np.pi * epoch_of_periastron_unix) / period
        
        return mean_anomaly % (2 * np.pi)

    def calculate_eccentric_anomaly(self, time):
        # Calculate the mean anomaly
        mean_anomaly = self.mean_angular_velocity() * time + self.mean_anomaly_at_unix
        mean_anomaly = mean_anomaly % (2 * np.pi)  # Ensure it is between 0 and 2π

        # Solve the Kepler equation to find the true anomaly
        eccentric_anomaly = self.solve_kepler_equation(mean_anomaly)
        return eccentric_anomaly

    def mean_angular_velocity(self):
        # Calculate the mean angular velocity (radians/second)
        # Using Kepler's third law T^2 = a^3
        period = torch.sqrt(self.semi_major_axis**3) * cons._year_to_second  # Orbital period (seconds)
        return 2 * np.pi / period

    def solve_kepler_equation(self, mean_anomaly):
        # Solve the Kepler equation M = E - e * sin(E) using Newton's method
        ecc_np = self.eccentricity.cpu().detach().numpy()
        ano_np = mean_anomaly.cpu().detach().numpy()
        def kepler_equation(E):
            return E - ecc_np * np.sin(E) - ano_np

        def kepler_equation_derivative(E):
            return 1 - ecc_np * np.cos(E)

        eccentric_anomaly = newton(kepler_equation, ano_np, fprime=kepler_equation_derivative)
        return torch.tensor(eccentric_anomaly)

    def calculate_position(self, eccentric_anomaly):
        # Calculate the position of the planet in the elliptical orbit
        # Convert from polar to Cartesian coordinates
        r = self.semi_major_axis * (1 - self.eccentricity * torch.cos(eccentric_anomaly))
        true_anomaly = 2 * torch.atan(torch.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)) * torch.tan(eccentric_anomaly / 2))

        x_prime = r * torch.cos(true_anomaly)
        y_prime = r * torch.sin(true_anomaly)

        # Trigonometric functions
        # Longitude of the ascending node (Omega)
        cos_lon = torch.cos(self.longitude_of_ascending_node)
        sin_lon = torch.sin(self.longitude_of_ascending_node)

        # Argument of periastron (omega)
        cos_arg = torch.cos(self.argument_of_periastron)
        sin_arg = torch.sin(self.argument_of_periastron)

        # Orbital inclination (i)
        cos_inc = torch.cos(self.inclination)
        sin_inc = torch.sin(self.inclination)

        # Planet coordinates relative to the host star
        x = (cos_lon*cos_arg - sin_lon*sin_arg*cos_inc) * x_prime - (cos_lon*sin_arg + sin_lon*cos_arg*cos_inc) * y_prime
        y = (sin_lon*cos_arg + cos_lon*sin_arg*cos_inc) * x_prime - (sin_lon*sin_arg - cos_lon*cos_arg*cos_inc) * y_prime
        z = (sin_arg*sin_inc) * x_prime + (cos_arg*sin_inc) * y_prime

        # Calculate the planet's celestial coordinates
        x_real = x + self.host_pos[0]
        y_real = y + self.host_pos[1]
        z_real = z + self.host_pos[2]

        # Project onto the celestial plane
        #ra = (torch.atan2(y_real, x_real)) * cons._radian_to_mas
        #dec = (torch.asin(z_real / torch.sqrt(x_real**2 + y_real**2 + z_real**2))) * cons._radian_to_mas
        ra = (torch.atan2(y_real, x_real) - self.target_longitude) * cons._radian_to_mas
        dec = (torch.asin(z_real / torch.sqrt(x_real**2 + y_real**2 + z_real**2)) - self.target_latitude) * cons._radian_to_mas
        return ra, dec

class GoatHerdEllipticalOrbit(EllipticalOrbit):
    '''
    Refer to arXiv:2103.15829
    '''
    def __init__(self):
        '''
        TODO: mean_anomaly_at_unix cause NaN or inf value in amplitude evaluation when it reaches the boundaries (0 or 2*pi).
              Very strange. Need to be fixed.
        '''
        super().__init__()
        self.register_buffer('N_it', torch.tensor(10, dtype=int))  # 迭代次数

    def solve_kepler_equation(self, mean_anomaly):
        #eccentric_anomaly = torch.where(self.eccentricity == 0., mean_anomaly, self.compute_contour(mean_anomaly))
        #return eccentric_anomaly
        """
        求解开普勒方程
    
        Args:
            mean_anomaly: 平近点角
        Returns:
            eccentric_anomaly: 偏心近点角
        """
        # 处理特殊情况
        zero_mask = torch.isclose(mean_anomaly, torch.tensor(0.0), atol=1e-10)
        two_pi_mask = torch.isclose(mean_anomaly, 2*torch.pi*torch.ones_like(mean_anomaly), atol=1e-10)
    
        # 对于e=0的情况,E=M
        circular_mask = self.eccentricity == 0.
    
        # 初始化结果tensor
        eccentric_anomaly = mean_anomaly.clone()
    
        # 处理特殊值
        eccentric_anomaly[zero_mask] = 0.0
        eccentric_anomaly[two_pi_mask] = 2*torch.pi
    
        # 对其他值使用contour方法求解
        normal_mask = ~(zero_mask | two_pi_mask | circular_mask)
        if normal_mask.any():
            eccentric_anomaly[normal_mask] = self.compute_contour(mean_anomaly[normal_mask])

        return eccentric_anomaly
    
    def compute_contour(self, mean_anomaly):
        """Solve Kepler's equation, E - e sin E = ell, via the contour integration method of Philcox et al. (2021)
        This uses techniques described in Ullisch (2020) to solve the `geometric goat problem'.

        Args:
            mean_anomaly (np.ndarray): Array of mean anomalies, ell, in the range (0,2 pi).
            eccentricity (float): Eccentricity. Must be in the range 0<e<1.
            N_it (float): Number of grid-points.

        Returns:
            np.ndarray: Array of eccentric anomalies, E.
        """

        # Check inputs
        if self.eccentricity<0.:
            raise Exception("Eccentricity must be greater than zero!")
        elif self.eccentricity>=1:
            raise Exception("Eccentricity must be less than unity!")
        if torch.max(mean_anomaly)>2.*torch.pi:
            raise Exception("Mean anomaly should be in the range (0, 2 pi)")
        if torch.min(mean_anomaly)<0:
            raise Exception("Mean anomaly should be in the range (0, 2 pi)")
        if self.N_it<2:
            raise Exception("Need at least two sampling points!")

        # Define sampling points
        N_points = self.N_it - 2
        N_fft = (self.N_it-1)*2

        # Define contour radius
        radius = self.eccentricity/2

        # Generate e^{ikx} sampling points and precompute real and imaginary parts
        j_arr = torch.arange(N_points)
        #freq = (2*torch.pi*(j_arr+1.)/N_fft)[:,torch.newaxis]
        freq = (2 * torch.pi * (j_arr + 1.) / N_fft)[:, None]
        exp2R = torch.cos(freq)
        exp2I = torch.sin(freq)
        ecosR= self.eccentricity*torch.cos(radius*exp2R)
        esinR = self.eccentricity*torch.sin(radius*exp2R)
        exp4R = exp2R*exp2R-exp2I*exp2I
        exp4I = 2.*exp2R*exp2I
        coshI = torch.cosh(radius*exp2I)
        sinhI = torch.sinh(radius*exp2I)

        # Precompute e sin(e/2) and e cos(e/2)
        esinRadius = self.eccentricity*torch.sin(radius)
        ecosRadius = self.eccentricity*torch.cos(radius)

        # Define contour center for each ell and precompute sin(center), cos(center)
        filt = mean_anomaly<torch.pi
        center = mean_anomaly-self.eccentricity/2.
        center[filt] = center[filt] + self.eccentricity
        sinC = torch.sin(center)
        cosC = torch.cos(center)
        output = center

        ## Accumulate Fourier coefficients
        # NB: we halve the integration range by symmetry, absorbing factor of 2 into ratio

        ## Separate out j = 0 piece, which is simpler

        # Compute z in real and imaginary parts (zI = 0 here)
        zR = center + radius

        # Compute e*sin(zR) from precomputed quantities
        tmpsin = sinC*ecosRadius+cosC*esinRadius

        # Compute f(z(x)) in real and imaginary parts (fxI = 0)
        fxR = zR - tmpsin - mean_anomaly
        eps = 1e-10
        fxR = torch.where(torch.abs(fxR) < eps, eps * torch.sign(fxR), fxR)

         # Add to arrays, with factor of 1/2 since an edge
        ft_gx2 = 0.5/fxR
        ft_gx1 = 0.5/fxR

        ## Compute j = 1 to N_points pieces

        # Compute z in real and imaginary parts
        zR = center + radius*exp2R
        zI = radius*exp2I

        # Compute f(z(x)) in real and imaginary parts
        # can use precomputed cosh / sinh / cos / sin for this!
        tmpsin = sinC*ecosR+cosC*esinR # e sin(zR)
        tmpcos = cosC*ecosR-sinC*esinR # e cos(zR)

        fxR = zR - tmpsin*coshI-mean_anomaly
        fxI = zI - tmpcos*sinhI

        # Compute 1/f(z) and append to array
        ftmp = fxR*fxR+fxI*fxI
        ftmp = torch.where(ftmp < eps, eps, ftmp)
        fxR = fxR /ftmp
        fxI = fxI /ftmp

        ft_gx2 = ft_gx2 + torch.sum(exp4R*fxR+exp4I*fxI,axis=0)
        ft_gx1 = ft_gx1 + torch.sum(exp2R*fxR+exp2I*fxI,axis=0)

        ## Separate out j = N_it piece, which is simpler

        # Compute z in real and imaginary parts (zI = 0 here)
        zR = center - radius

        # Compute sin(zR) from precomputed quantities
        tmpsin = sinC*ecosRadius-cosC*esinRadius

        # Compute f(z(x)) in real and imaginary parts (fxI = 0 here)
        fxR = zR - tmpsin-mean_anomaly

        # Add to sum, with 1/2 factor for edges
        ft_gx2 = ft_gx2 + 0.5/fxR
        ft_gx1 = ft_gx1 + -0.5/fxR

        ### Compute and return the solution E(ell,e)
        output = output + radius*ft_gx2/ft_gx1

        output = torch.where(
            torch.isnan(output) | torch.isinf(output),
            mean_anomaly,  # 如果计算失败则返回输入值
            output
        )

        return output