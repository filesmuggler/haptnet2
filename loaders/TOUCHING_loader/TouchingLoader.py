import yaml
import scipy.io
import numpy as np

class TOUCHING_Loader:
    def __init__(self, path_to_dataset=None,pick_modalities=None, save_to_pickle=False):
        self.path_to_dataset = path_to_dataset
        self._read_dataset()

    def _read_dataset(self):
        self.dataset = scipy.io.loadmat(self.path_to_dataset)

if __name__ == "__main__":
    config_file = "./config/touching.yaml"

    with open(config_file, 'r') as file_yaml:
        touching_config = yaml.safe_load(file_yaml)

    t_loader = TOUCHING_Loader(
        path_to_dataset=touching_config['path_to_dataset'],
        pick_modalities=touching_config['modalities'],
        save_to_pickle=touching_config['save_to_pickle']
    )

    print("finished")
