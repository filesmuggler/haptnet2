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