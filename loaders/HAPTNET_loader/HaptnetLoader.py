import pickle
import yaml
import numpy as np

class HAPTNET_Loader:
    def __init__(self, path_to_dataset=None,pick_modalities=None, save_to_pickle=False):
        self.path_to_data = path_to_dataset
        self.modalities = pick_modalities
        self._read_dataset()
        self._pick_modalities()
        # self._filter_dataset()

    def _read_dataset(self):
        infile = open(self.path_to_data, 'rb')
        self.dataset = pickle.load(infile)
        self.stats = self.dataset['stats']
        self.classes = self.dataset['classes']
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


    def _pick_modalities(self):
        filtered_train = []
        filtered_test = []
        for data_entry in self.dataset['train']:
            temp_list = []
            temp_list.append(data_entry[0])

            self.modalities_filtered = []
            for modality in self.modalities:
                if self.modalities[modality]:
                    self.modalities_filtered.append(modality)

            for modality in self.modalities:
                if modality == "force" and self.modalities[modality]:
                    temp_list.append(data_entry[1])
                if modality == "imu0" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 0:4])
                if modality == "imu1" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 4:8])
                if modality == "imu2" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 8:12])
                if modality == "imu3" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 12:16])
            filtered_train.append(tuple(temp_list))

        for data_entry in self.dataset['test']:
            temp_list = []
            temp_list.append(data_entry[0])
            for modality in self.modalities:
                if modality == "force" and self.modalities[modality]:
                    temp_list.append(data_entry[1])
                if modality == "imu0" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 0:4])
                if modality == "imu1" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 4:8])
                if modality == "imu2" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 8:12])
                if modality == "imu3" and self.modalities[modality]:
                    temp_list.append(data_entry[2][:, 12:16])
            filtered_test.append(tuple(temp_list))

        self.dataset_train_filtered = filtered_train
        self.dataset_test_filtered = filtered_test

    def get_train_dataset(self):
        return self.dataset_train_filtered

    def get_test_dataset(self):
        return self.dataset_test_filtered

    def get_modalities(self):
        return self.modalities_filtered

if __name__ == "__main__":
    config_file = "./config/hapnet.yaml"

    with open(config_file, 'r') as file_yaml:
        hapnet_config = yaml.safe_load(file_yaml)

    h_loader = HAPTNET_Loader(
        path_to_dataset=hapnet_config['path_to_dataset'],
        pick_modalities=hapnet_config['modalities'],
        save_to_pickle=hapnet_config['save_to_pickle']
    )

    print("finished")