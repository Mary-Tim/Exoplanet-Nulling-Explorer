import sys
sys.path.append('../..')

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda:0')

import numpy as np
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)
from tqdm import tqdm

from nullingexplorer.utils import get_transmission
from nullingexplorer import *
from nullingexplorer.utils import Constants as cons
from nullingexplorer.utils import Configuration as cfg
from nullingexplorer.generator import AmplitudeCreator, ObservationCreator

from itertools import cycle
cycol = cycle('bgrcmk')

fov = np.array([0, 120.], dtype=np.float64) / cons._radian_to_mas
bins = 200

distance = 4.2 * cons._light_year_to_meter
star_radius = 0.1542*cons._sun_radius*1e3
star_mas = star_radius / distance * cons._radian_to_mas

poxb_mas = 0.0485*cons._au_to_meter / (4.2 * cons._light_year_to_meter) * cons._radian_to_mas
poxd_mas = 0.02885*cons._au_to_meter / (4.2 * cons._light_year_to_meter) * cons._radian_to_mas

resolution = 2

print(poxb_mas)
print(poxd_mas)

obs_config = {
    'Spectrum':{
        'Type': 'Equal',
        'BinNumber': 1,
        'Low': 4,
        'High': 4,        # unit: micrometer
    },
    'Observation':{
        'ObsNumber': 1.,
        'IntegrationTime': 1.,  # unit: second
        'ObsMode': [1],  # [1] or [-1] or [1, -1]
        'Phase':{
            'Start' : 100.,
            'Stop': 100.,   # unit: degree
        },
        'Baseline':{
            'Type': 'Constant',
            'Value': 20.,  # unit: meter
        },
    },
    'Configuration':{
        # Formation parameters
        'ratio':    6,          # ratio of imaging baseline versus nulling baseline [dimensionless]]
        'formation_longitude': 0.,  # Formation longitude [degree] 
        'formation_latitude' : 0.,  # Formation latitude [degree] 
        # Instrument parameters
        'mirror_diameter': 0.6,   # Diameter of MiYin primary mirror [meter]
        'quantum_eff': 0.7,     # Quantum efficiency of detector [dimensionless]
        'instrument_eff': 0.05, # Instrument throughput efficiency [dimensionless]
        'nulling_depth': 0.,    # Nulling depth of the instrument [dimensionless, within [0,1) ]
    }
}

model = get_transmission('SingleBracewell')()
obs_creator = ObservationCreator()

ra = torch.tensor(np.linspace(fov[0], fov[1], bins))
dec = torch.tensor(0.)

ra_numpy = ra.cpu().detach().numpy()

def draw_a_point(ax, resolution):

    wl_mid = 1.2e-6
    wl_lo = wl_mid - wl_mid/(resolution*2)
    wl_hi = wl_mid + wl_mid/(resolution*2)

    # Observation Config
    obs_config['Observation']['Phase']['Start'] = 0.
    obs_config['Observation']['Phase']['Stop']  = 0.
    obs_creator.load(obs_config)
    data = obs_creator.generate()
    data['wl_lo'] = torch.tensor([wl_lo])
    data['wl_hi'] = torch.tensor([wl_hi])

    #trans_map = model(ra_grid, dec_grid, data['wl_mid'], data).cpu().detach().numpy()
    wl_interp = torch.linspace(wl_lo, wl_hi, 1000)
    trans_interp_list = []
    for wl_mid in wl_interp:
        trans_interp_list.append(model(ra, dec, wl_mid, data))

    trans_interp_result = torch.stack(trans_interp_list)
    
    trans_map = torch.mean(trans_interp_result, dim=0).cpu().detach().numpy()

    #draw_trans = griddata(points.cpu().detach().numpy(), trans_map.flatten(), (ra_grid_numpy, dec_grid_numpy), method='cubic')

    levels = np.arange(0., 1.0001, 0.01)
    color = next(cycol)
    trans_map_cont = ax.plot(ra_numpy*cons._radian_to_mas, trans_map, color=color, label=f"R: {resolution:d}")



    #plt.savefig(f"results/trans_map_show/trans_map_{i}.jpg")
    #plt.close()


if __name__ == '__main__':
    phase_num = 360
    phase_list = np.linspace(0., 360., phase_num, endpoint=False)
    #for i, phase in tqdm(enumerate(phase_list), total=phase_num):
    #    draw_a_point(i, phase)
    fig, ax = plt.subplots()
    ax.axvline(x=poxb_mas, color='b', linestyle='--', label='Proxima b')
    ax.axvline(x=poxd_mas, color='g', linestyle='--', label='Proxima d')
    for res in [2, 5, 10, 20, 100]:
        draw_a_point(ax, res)
    ax.set_xlabel("Distance [mas]")
    ax.set_ylabel("Transmission")
    ax.legend()
    plt.show()
