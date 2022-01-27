"""Haptnet_Preprocessor class"""
__docformat__ = "google"

import pickle
import yaml
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

class Haptnet_Preprocessor:
    def __init__(self,path_to_config: str):
        """
        Haptnet_Preprocessor prepares raw dataset for training according to parameters
        specified in the yaml config file in package's config folder
        Args:
            path_to_config: yaml file with parameters for preprocessing
        """
        self.config_path = path_to_config

        with open(self.config_path, 'r') as file_yaml:
            self.config = yaml.safe_load(file_yaml)


    def load_data_from_path(self, path_to_file: str) -> list:
        """
        Loads dataset
        Args:
            path_to_file: path to pickle or pkl file with dataset

        Returns:
            dataset in the form of the list
        """
        dataset_file = open(path_to_file, "rb")
        dataset = pickle.load(dataset_file)
        return dataset

    def resample_data(self,raw_inputs: list, max_len: int=0) -> list:
        """
        resample
        Args:
            raw_inputs:

        Returns:

        """
        if max_len == 0:
            maxlen = self._find_max_num_samples(raw_inputs)
            max_max = max(maxlen)
        else:
            max_max = max_len
        resampled_dataset = []

        # TODO: assumption the first element is class
        # other elements are datastreams
        for element in raw_inputs:
            temp_array = []
            temp_array.append(element[0])
            for i, data_stream in enumerate(element[1:]):
                data_stream_r = resample(data_stream,max_max)
                temp_array.append(data_stream_r)

            resampled_dataset.append(tuple(temp_array))

        return resampled_dataset


    def _find_max_num_samples(self,dataset: list) -> list:
        """

        Args:
            dataset: assume first element of every tuple is class descriptor
                     any other element is data stream
        Returns:

        """

        # get representative element from dataset
        data_stream_len = len(dataset[0][1:])

        maximal = data_stream_len*[0]

        for element in dataset:
            for i,data_stream in enumerate(element[1:]):
                if len(data_stream) > maximal[i]:
                    maximal[i] = len(data_stream)

        return maximal

    def add_labels(self,dataset:list)->list:
        """

        Args:
            config:
            dataset:

        Returns:

        """
        new_dataset = []

        for data_entry in tqdm(dataset):
            y = data_entry[0]
            new_y = []

            for i, label_cat in enumerate(self.config['one_hot_classes']):
                if y==label_cat:
                    new_y.append(self.config['one_hot_classes'][label_cat])

            new_dataset.append((new_y, data_entry[1],data_entry[2]))

        return new_dataset

    def preprocess_data(self)-> dict:
        """
        Combines all preprocessing techinques from config file
        Returns:
            preprocessed dataset
        """
        #read
        train_dataset = self.load_data_from_path(self.config["raw_dataset_train"])
        test_dataset = self.load_data_from_path(self.config["raw_dataset_test"])

        #resample
        #TODO: not tested against additional modalities like acc, gyro
        if self.config["resample"]:
            train_dataset = self.resample_data(train_dataset, self.config['new_number_of_samples'])
            test_dataset = self.resample_data(test_dataset, self.config['new_number_of_samples'])

        ## standarize forces, acc, gyro, not quaternions - quats are already zero mean, unit variance by default
        # mean and std train
        # TODO: not tested against additional modalities like acc, gyro
        if self.config["standarization"]:
            self.mean_fx = np.mean([np.mean(train_dataset[i][1][:,0]) for i in range(len(train_dataset))])
            self.mean_fy = np.mean([np.mean(train_dataset[i][1][:,1]) for i in range(len(train_dataset))])
            self.mean_fz = np.mean([np.mean(train_dataset[i][1][:,2]) for i in range(len(train_dataset))])

            self.std_fx = np.mean([np.std(train_dataset[i][1][:, 0]) for i in range(len(train_dataset))])
            self.std_fy = np.mean([np.std(train_dataset[i][1][:, 1]) for i in range(len(train_dataset))])
            self.std_fz = np.mean([np.std(train_dataset[i][1][:, 2]) for i in range(len(train_dataset))])

            new_train_dataset = []
            for data_entry in train_dataset:
                forces_x = (data_entry[1][:, 0] - self.mean_fx) / self.std_fx
                forces_y = (data_entry[1][:, 1] - self.mean_fy) / self.std_fy
                forces_z = (data_entry[1][:, 2] - self.mean_fz) / self.std_fz
                forces = np.stack((forces_x, forces_y, forces_z), axis=-1)

                new_train_dataset.append((data_entry[0], forces, data_entry[2]))

            train_dataset = new_train_dataset

            new_test_dataset = []
            for data_entry in test_dataset:
                forces_x = (data_entry[1][:, 0] - self.mean_fx) / self.std_fx
                forces_y = (data_entry[1][:, 1] - self.mean_fy) / self.std_fy
                forces_z = (data_entry[1][:, 2] - self.mean_fz) / self.std_fz
                forces = np.stack((forces_x, forces_y, forces_z), axis=-1)

                new_test_dataset.append((data_entry[0], forces, data_entry[2]))

            test_dataset = new_test_dataset

        if self.config['one_hot_encoding']:
            train_dataset = self.add_labels(train_dataset)
            test_dataset = self.add_labels(test_dataset)

        #complete info
        complete_dataset = {}
        complete_dataset["stats"] = {
            "mean_fx":self.mean_fx,
            "mean_fy":self.mean_fy,
            "mean_fz":self.mean_fz,
            "std_fx":self.std_fx,
            "std_fy":self.std_fy,
            "std_fz":self.std_fz,
        }
        complete_dataset['classes']=self.config['one_hot_classes']
        complete_dataset['train']=train_dataset
        complete_dataset['test']=test_dataset
        return complete_dataset



if __name__ == "__main__":
    config_file = "./config/config.yaml"

    h_prep = Haptnet_Preprocessor(config_file)

    #train_raw = h_prep.load_data_from_path(h_prep.config["raw_dataset_train"])
    new_prep_data = h_prep.preprocess_data()

    # save new dataset
    with open(h_prep.config["preprocessed_dataset"], 'wb') as handle:
        pickle.dump(new_prep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("finished")