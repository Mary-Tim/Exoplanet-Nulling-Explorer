import torch

from nullingexplorer.generator import ObservationCreator
from nullingexplorer.generator import AmplitudeCreator
from nullingexplorer.io import DataHandler

class PoissonSignificance():
    def __init__(self) -> None:
        self.obs_creator = ObservationCreator()
        self.__obs_config = None
        self.__sig_amp_config = None
        self.__bkg_amp_config = None

    @property
    def obs_config(self):
        return self.__obs_config
    
    @obs_config.setter
    def obs_config(self, config):
        self.__obs_config = config

    @property
    def sig_amp_config(self):
        return self.__sig_amp_config
    
    @sig_amp_config.setter
    def sig_amp_config(self, config):
        self.__sig_amp_config = config

    @property
    def bkg_amp_config(self):
        return self.__bkg_amp_config
    
    @bkg_amp_config.setter
    def bkg_amp_config(self, config):
        self.__bkg_amp_config = config
    
    def get_significance(self, sig_pe: torch.Tensor, bkg_pe: torch.Tensor):
        def cal_SNR_wl(sig, bg):
            #return torch.sum(torch.sqrt(sig**2)) / torch.sqrt(torch.sum(bg) + torch.sum(torch.sqrt(sig**2)))
            return torch.sum(torch.sqrt(sig**2)) / torch.sqrt(2 * torch.sum(bg) + torch.sum(torch.sqrt(sig**2)))

        SNR_wl = torch.vmap(cal_SNR_wl)(sig_pe, bkg_pe)
        SNR = torch.sqrt(torch.sum(SNR_wl**2))
        return SNR

    def gen_sig_pe(self, sig_config = None):
        if sig_config is not None:
            self.__sig_amp_config = sig_config

        if self.__obs_config is None:
            raise ValueError('No observation config')
        if self.__sig_amp_config is None:
            raise ValueError('No signal amplitude config')

        self.__obs_config['Observation']['ObsMode'] = [1, -1]
        self.obs_creator.load(self.__obs_config)
        sig_data = self.obs_creator.generate()

        sig_amp = AmplitudeCreator(config=self.__sig_amp_config)
        sig_data['photon_electron'] = sig_amp(sig_data)
        data_handler = DataHandler(sig_data)
        sig_data = data_handler.diff_data(self.obs_creator)

        sig_pe = sig_data['photon_electron'].reshape(self.obs_creator.obs_num, self.obs_creator.spec_num).t()
        return sig_pe

    def gen_bkg_pe(self, bkg_config = None):
        if bkg_config is not None:
            self.__bkg_amp_config = bkg_config

        if self.__obs_config is None:
            raise ValueError('No observation config')
        if self.__bkg_amp_config is None:
            raise ValueError('No background amplitude config')

        self.__obs_config['Observation']['ObsMode'] = [1]
        self.obs_creator.load(self.__obs_config)
        bkg_data = self.obs_creator.generate()

        bkg_amp = AmplitudeCreator(config=self.__bkg_amp_config)
        bkg_data['photon_electron'] = bkg_amp(bkg_data)

        bkg_pe = bkg_data['photon_electron'].reshape(self.obs_creator.obs_num, self.obs_creator.spec_num).t()
        return bkg_pe