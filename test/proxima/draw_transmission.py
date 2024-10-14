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

fov = np.array([-75., 75.], dtype=np.float64) / cons._radian_to_mas
bins = 200

distance = 4.2 * cons._light_year_to_meter
star_radius = 0.1542*cons._sun_radius*1e3
star_mas = star_radius / distance * cons._radian_to_mas

poxb_mas = 0.0485*cons._au_to_meter / (4.2 * cons._light_year_to_meter) * cons._radian_to_mas
poxd_mas = 0.02885*cons._au_to_meter / (4.2 * cons._light_year_to_meter) * cons._radian_to_mas

print(poxb_mas)
print(poxd_mas)

obs_config = {
    'Spectrum':{
        'Type': 'Equal',
        'BinNumber': 1,
        'Low': 1.2,
        'High': 1.2,        # unit: micrometer
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
            'Value': 3.,  # unit: meter
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
dec = torch.tensor(np.linspace(fov[0], fov[1], bins))

ra_grid, dec_grid = torch.meshgrid(ra, dec, indexing='ij')
ra_grid_numpy = ra_grid.cpu().detach().numpy()
dec_grid_numpy = dec_grid.cpu().detach().numpy()

points = torch.stack((ra_grid.flatten(), dec_grid.flatten()), -1)

def draw_a_point(i, phase):

    # Observation Config
    obs_config['Observation']['Phase']['Start'] = phase
    obs_config['Observation']['Phase']['Stop']  = phase
    obs_creator.load(obs_config)
    data = obs_creator.generate()

    trans_map = model(ra_grid, dec_grid, data['wl_mid'], data).cpu().detach().numpy()

    #draw_trans = griddata(points.cpu().detach().numpy(), trans_map.flatten(), (ra_grid_numpy, dec_grid_numpy), method='cubic')

    fig, ax = plt.subplots()
    levels = np.arange(0., 1.0001, 0.01)
    trans_map_cont = ax.contourf(ra_grid_numpy*cons._radian_to_mas, dec_grid_numpy*cons._radian_to_mas, trans_map, levels=levels, cmap = plt.get_cmap("gray"))
    ax.set_xlabel("RA [mas]")
    ax.set_ylabel("DEC [mas]")

    sun = plt.Circle((0., 0.), star_mas, color='r')
    orbitb = plt.Circle((0., 0.), poxb_mas, color='b', fill=False)
    orbitd = plt.Circle((0., 0.), poxd_mas, color='g', fill=False)

    ax.set_aspect(1)
    ax.add_artist(sun)
    ax.add_artist(orbitb)
    ax.add_artist(orbitd)

    cbar = fig.colorbar(trans_map_cont)

    plt.show()
    #plt.savefig(f"results/trans_map_show/trans_map_{i}.jpg")
    #plt.close()


if __name__ == '__main__':
    phase_num = 360
    phase_list = np.linspace(0., 360., phase_num, endpoint=False)
    #for i, phase in tqdm(enumerate(phase_list), total=phase_num):
    #    draw_a_point(i, phase)
    draw_a_point(0, 0.)
