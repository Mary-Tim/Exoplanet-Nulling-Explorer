import numpy as np
class Constants(object):
    #_device = 'cuda:0'                         # Default device
    _Planck_constant = 6.62607015e-34           # Planck constant (unit: J/Hz)
    _light_speed = 299792458.                   # speed of light (unit: m/s)
    _Boltzmann_constant = 1.380649e-23          # Boltzmann constant (unit: m^2 kg s^{-2} K^{-1})

    _pc_to_meter = 3.0856775814913673e16        # one parsec (unit: meter)
    _au_to_meter = 149597870700.                # Uit concersion, from AU to meter (unit: AU)
    _pc_to_au = _pc_to_meter / _au_to_meter     # Uit concersion, from parsec to AU (unit: parsec)
    _radian_to_mas = 180. / np.pi * 3600. * 1e3 # Uit concersion, from radian to mac (unit: radian)
    _radian_to_degree = 180. / np.pi            # Uit concersion, from radian to degree (unit: radian)
    _light_year_to_meter = 9.4605284e15         # Uit concersion, from light year to meter (unit: light year)
    _year_to_second = 365.2425 * 24. * 3600.    # Uit concersion, from year to second (unit: year)

    _sun_radius = 695500.                       # Radius of the sun (unit: kilometer)
    _earth_radius = 6371.e3                     # Radius of the earth (unit: meter)  
