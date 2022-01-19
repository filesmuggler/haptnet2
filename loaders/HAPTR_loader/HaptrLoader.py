import pickle
import yaml
import numpy as np

class HAPTR_Loader:
    def __init__(self, path_to_dataset=None,pick_modalities=None, save_to_pickle=False):
        self.path_to_data = path_to_dataset
        self.modalities = pick_modalities
        self._read_dataset()
        self._filter_dataset()

    def _read_dataset(self):
        infile = open(self.path_to_data, 'rb')
        self.dataset = pickle.load(infile)
        infile.close()

    def _filter_dataset(self):
        extracted_signals_train = np.array([self.dataset['train_ds'][i]['signal'][:, self.modalities] for i in range(len(self.dataset['train_ds']))])
        extracted_labels_train = [self.dataset['train_ds'][i]['label'] for i in range(len(self.dataset['train_ds']))]

        extracted_signals_val = np.array([self.dataset['val_ds'][i]['signal'][:, self.modalities] for i in range(len(self.dataset['val_ds']))])
        extracted_labels_val = [self.dataset['val_ds'][i]['label'] for i in range(len(self.dataset['val_ds']))]

        extracted_signals_test = np.array([self.dataset['test_ds'][i]['signal'][:, self.modalities] for i in range(len(self.dataset['test_ds']))])
        extracted_labels_test = [self.dataset['test_ds'][i]['label'] for i in range(len(self.dataset['test_ds']))]

        extracted_signals_norm_train = [(extracted_signals_train[i] - np.mean(extracted_signals_train[i])) / np.std(extracted_signals_train[i]) for i in range(len(extracted_signals_train))]
        extracted_signals_norm_val = [(extracted_signals_val[i] - np.mean(extracted_signals_val[i])) / np.std(extracted_signals_val[i]) for i in range(len(extracted_signals_val))]
        extracted_signals_norm_test = [(extracted_signals_test[i] - np.mean(extracted_signals_test[i])) / np.std(extracted_signals_test[i]) for i in range(len(extracted_signals_test))]

        self.extracted_dataset_train = [(extracted_labels_train[i], extracted_signals_train[i]) for i in range(len(extracted_signals_train))]
        self.extracted_dataset_val = [(extracted_labels_val[i], extracted_signals_val[i]) for i in range(len(extracted_signals_val))]
        self.extracted_dataset_test = [(extracted_labels_test[i], extracted_signals_test[i]) for i in range(len(extracted_signals_test))]

        self.extracted_dataset_norm_train = [(extracted_labels_train[i], extracted_signals_norm_train[i]) for i in
                                        range(len(extracted_signals_norm_train))]
        self.extracted_dataset_norm_val = [(extracted_labels_val[i], extracted_signals_norm_val[i]) for i in
                                      range(len(extracted_signals_norm_val))]
        self.extracted_dataset_norm_test = [(extracted_labels_test[i], extracted_signals_norm_test[i]) for i in
                                       range(len(extracted_signals_norm_test))]


if __name__ == "__main__":
    config_file = "./config/haptr.yaml"

    with open(config_file, 'r') as file_yaml:
        haptr_config = yaml.safe_load(file_yaml)

    h_loader = HAPTR_Loader(
        path_to_dataset=haptr_config['path_to_dataset'],
        pick_modalities=haptr_config['modalities'],
        save_to_pickle=haptr_config['save_to_pickle']
    )

    print("finished")