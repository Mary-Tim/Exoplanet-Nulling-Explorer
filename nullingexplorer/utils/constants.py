import numpy as np
class Constants(object):
    _device = 'cuda:0'                          # Default device
    _Planck_constant = 6.62607015e-34           # Planck constant (unit: J/Hz)
    _light_speed = 299792458.                   # speed of light (unit: m/s)
    _Boltzmann_constant = 1.380649e-23          # Boltzmann constant (unit: m^2 kg s^{-2} K^{-1})

    _pc_to_meter = 3.0856775814913673e16        # one parsec (unit: meter)
    _radian_to_mac = 180. / np.pi * 3600. * 1e3 # Uit concersion, from radian to mac (unit: radian)
    _radian_to_degree = 180. / np.pi            # Uit concersion, from radian to degree (unit: radian)
    _au_to_meter = 149597870700.                # Uit concersion, from AU to meter (unit: AU)

