import scipy.io
import scipy.stats
import scipy

import yaml


class STINK_Loader:
    def __init__(self,path_to_dataset=None, pick_modalities=None, save_to_pickle=False):
        # dataset is already standarized with zscore
        self.dataset = scipy.io.loadmat(path_to_dataset)
        self.num_classes = None
        self.mean = None
        self.std = None

        self.pick_modalities = pick_modalities
        if self.pick_modalities != None:
            self.num_modalities = len(self.pick_modalities)

        self._numpify_dataset()

    def _numpify_dataset(self):
        extracted_signals = [self.dataset['sensor_readings'][i][0][:, self.pick_modalities]
             for i in range(len(self.dataset['sensor_readings']))]

        extracted_labels = self.dataset['label_vector'][:,1]

        self.extracted_dataset = [(extracted_labels[i],extracted_signals[i])
                                  for i in range(len(extracted_signals))]


    def get_original_dataset(self):
        return self.dataset

    def get_numpified_dataset(self):
        return self.extracted_dataset

    def get_dataset_parameters(self):
        signals_mean = [scipy.mean(self.dataset['sensor_readings'][i][0],0)
                          for i in range(len(self.dataset['sensor_readings']))]
        self.mean = scipy.mean(signals_mean)
        print("ORIGINAL STINK mean: ", self.mean)

        signals_std = [scipy.std(self.dataset['sensor_readings'][i][0],0)
                       for i in range(len(self.dataset['sensor_readings']))]
        self.std = scipy.mean(signals_std)
        print("ORIGINAL STINK std: ", self.std)


if __name__ == "__main__":
    config_file = "./config/stink.yaml"

    with open(config_file,'r') as file_yaml:
        stink_config = yaml.safe_load(file_yaml)

    s_loader = STINK_Loader(
                            path_to_dataset=stink_config['path_to_dataset'],
                            pick_modalities=stink_config['modalities'],
                            save_to_pickle=stink_config['save_to_pickle']
                            )
    stink_dataset = s_loader.get_original_dataset()
    s_loader.get_dataset_parameters()
    print("finished")
