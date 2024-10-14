import torch
from tensordict import PersistentTensorDict
from tensordict import TensorDict

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.LHCb2)

from nullingexplorer.generator import ObservationCreator

class DataHandler():
    def __init__(self, data: TensorDict=None) -> None:
        self.__data = data

    #def __getattr__(self, name: str) -> torch.Any:
    #    return self.__data[name]

    #def __setattr__(self, name: str, value: torch.Any) -> None:
    #    self.__data[name] = value

    @property
    def data(self) -> TensorDict:
        return self.__data

    @data.setter
    def data(self, data: TensorDict) -> None:
        self.__data = data

    def save(self, path: str, file_name='data.hdf5'):
        if not path.endswith(('.hdf5', '.h5',)):
            if path.endswith('/'):
                path = f"{path}{file_name}"
            else:
                path = f"{path}/{file_name}"
                
        with open(path, "w") as file_h5:
            self.__data.detach().to_h5(file_h5.name, compression="gzip", compression_opts=9)
            file_h5.close()

    @classmethod
    def load(cls, path: str):
        data_h5 = PersistentTensorDict.from_h5(path)
        data = data_h5.to_tensordict()
        data_h5.close()
        return cls(data)

    def diff_data(self, obs_creator: ObservationCreator):
        if self.data == None:
            raise ValueError("Data is not loaded yet.")
        diff_data = self.data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)[:,:,0]
        data_mod3 = self.data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)[:,:,0]
        data_mod4 = self.data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)[:,:,1]
        diff_data['photon_electron'] = (data_mod3['photon_electron'] - data_mod4['photon_electron'])
        diff_data['pe_uncertainty'] = torch.sqrt(data_mod3['photon_electron'] + data_mod4['photon_electron'])
        diff_data['pe_uncertainty'][diff_data['pe_uncertainty'] == 0] = 1e10
        self.data = diff_data.flatten()
        return self.data

    def reshape(self, obs_creator: ObservationCreator):
        if self.data == None:
            raise ValueError("Data is not loaded yet.")
        if obs_creator.mod_num > 1:
            self.data = self.data.reshape(obs_creator.obs_num,obs_creator.spec_num,obs_creator.mod_num)
        else:
            self.data = self.data.reshape(obs_creator.obs_num,obs_creator.spec_num)
        return self.data
        
    def draw(self, obs_creator: ObservationCreator, draw_err=False):
        phase_number = obs_creator.obs_num
        wl_number = obs_creator.spec_num
        phase_bins = self.data['phase'].reshape(phase_number, wl_number).cpu().detach().numpy()
        wl_bins = self.data['wl_mid'].reshape(phase_number, wl_number).cpu().detach().numpy()

        def draw_data(ax, name, label=None):
            if label == None:
                label = name
            data = self.data[name].reshape(phase_number, wl_number)
            if name == 'pe_uncertainty':
                data[data == 1e10] = 0.
            cm_data = ax.pcolormesh(phase_bins, wl_bins, 
                                    data.cpu().detach().numpy(), 
                                    cmap = plt.get_cmap("bwr"))
            ax.set_xlabel("phase / rad")
            ax.set_ylabel("wavelength / m")
            fig.colorbar(cm_data, label=label, aspect=5, pad=0.01)

        if draw_err == False:
            fig, ax = plt.subplots(figsize=(24.,3.5))
            draw_data(ax, name='photon_electron', label="Photon electron")
        else:
            fig, ax = plt.subplots(2, 1, figsize=(24.,7.))
            draw_data(ax[0], name='photon_electron', label="Photon electron")
            draw_data(ax[1], name='pe_uncertainty', label="Uncertainty")

        plt.show()

    '''
    phase: torch.Tensor
    wavelength: torch.Tensor
    wl_width: torch.Tensor
    mod: torch.Tensor
    integral_time: torch.Tensor
    photon_electron: torch.Tensor
    pe_uncertainty: torch.Tensor

    def select_mod(self, mod):
        return self[self.mod==mod]

    def select_data(self, key, val):
        return self[getattr(self, key)==val]

    def nparray(self, key):
        return getattr(self, key).cpu().detach().numpy()

    def get_bins(self, key):
        return torch.unique(getattr(self, key))

    def get_bin_number(self, key):
        return torch.tensor(len(torch.unique(getattr(self, key))))

    #def save(self, path: str):
    #    if not (path.endswith('.hdf5') or path.endswith('.h5')):
    #        path = path + '.hdf5'
    def save(self, path: str, file_name='data.hdf5'):
        if not path.endswith(('.hdf5', '.h5',)):
            if path.endswith('/'):
                path = f"{path}{file_name}"
            else:
                path = f"{path}/{file_name}"
                
        with open(path, "w") as file_h5:
            self.detach().to_h5(file_h5.name, compression="gzip", compression_opts=9)
            file_h5.close()

    def draw(self, draw_err=False, save=False, show=False, path='.', name='dataset_distribution'):
        phase_number = self.get_bin_number('phase')
        wl_number = self.get_bin_number('wavelength')
        phase_bins = self.phase.reshape(phase_number, wl_number).cpu().detach().numpy()
        wl_bins = self.wavelength.reshape(phase_number, wl_number).cpu().detach().numpy()

        def draw_data(ax, name, label=None):
            if label == None:
                label = name
            data = getattr(self, name).reshape(phase_number, wl_number)
            if name == 'pe_uncertainty':
                data[data == 1e10] = 0.
            cm_data = ax.pcolormesh(phase_bins, wl_bins, 
                                    data.cpu().detach().numpy(), 
                                    cmap = plt.get_cmap("bwr"))
            ax.set_xlabel("phase / rad")
            ax.set_ylabel("wavelength / m")
            fig.colorbar(cm_data, label=label, aspect=5, pad=0.01)

        if draw_err == False:
            fig, ax = plt.subplots(figsize=(24.,3.5))
            draw_data(ax, name='photon_electron', label="Photon electron")
        else:
            fig, ax = plt.subplots(2, 1, figsize=(24.,7.))
            draw_data(ax[0], name='photon_electron', label="Photon electron")
            draw_data(ax[1], name='pe_uncertainty', label="Uncertainty")

        if save == True:
            plt.savefig(f"{path}/{name}.pdf")
        if show == True:
            plt.show()

    @classmethod 
    def create_empty_dataset(cls, batch_size: list):
        return cls( phase=torch.zeros(batch_size), 
                    wavelength=torch.zeros(batch_size), 
                    wl_width=torch.zeros(batch_size), 
                    mod=torch.zeros(batch_size), 
                    integral_time=torch.zeros(batch_size),
                    photon_electron=torch.zeros(batch_size),
                    pe_uncertainty=torch.zeros(batch_size),
                    batch_size=batch_size
                    )

    @classmethod
    def load(cls, path: str):
        data_h5 = PersistentTensorDict.from_h5(path)
        data = cls.create_empty_dataset(batch_size=data_h5.batch_size)
        print(data.keys())
        for key in data.keys():
            setattr(data, key, data_h5[key])
        data_h5.close()
        return data
'''