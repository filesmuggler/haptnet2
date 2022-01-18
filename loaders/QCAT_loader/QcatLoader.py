# Adapted from https://data.csiro.au/collection/csiro:46885v2

# The dataset has two different sensor measurements from the quadruped robot DyRET:
# 4x force sensors on the feet (referred as raw) and an IMU sensor on the body (imu).
# The first number (0-5) shows the type to the terrain:
# 0. Concrete, 1. Grass, 2. Gravel, 3. Mulch, 4. Dirt, 5. Sand.
# Inside each file there are 10 trials: the forth column (third if starting from 0).
# The second number (1-6) represents the robot speed in each run.
# There were 8 steps for each trial, which means that we have: 6*10*6*8 = 2880 samples.
# The jupyter notebook code for processing the force sensor data and IMU data are provided.
# They give us 2880 force or IMU  samples.


import yaml
import  csv
import numpy as np

import pickle

class QCAT_Loader:
    def __init__(self, path_to_dataset=None,pick_modalities=None, save_to_pickle=False):
        self.path_to_data = path_to_dataset
        self.num_classes = 6  # we have 6 terrain classes
        self.num_trials = 10  # the robot walked on each terrain 10 times
        self.num_steps = 8  # the robot walked 8 steps on each terrain
        self.num_diff_speeds = 6
        self.max_steps = 662  # this is obtained based on our data
        self.all_colms = 16  # this is based on number of all colms in the csv files
        self.relevant_colms = 12  # this is our force sensor information in the csv files: (Forward Right (FR), FL, BR, BL: XYZ)
        self.all_seq = self.num_classes * self.num_diff_speeds * self.num_trials * self.num_steps

        self.all_data = np.zeros([self.all_seq, self.max_steps, self.all_colms])
        self.data_steps_array = np.zeros([self.all_seq, self.max_steps, self.relevant_colms])
        self.data_labels_array = np.zeros((self.all_seq, self.num_classes))
        self.data_length_array = np.zeros((self.all_seq))
        self.data_length_array = self.data_length_array.astype(int)

        self._read_dataset()

    def _read_dataset(self):
        cnt = 0
        for i in range(self.num_classes):
            for j in range(1, 7):  # different speeds
                tmp_data = []
                tmp_list = []
                struct = f'{self.path_to_data}/%d_%d_legSensors_raw.csv' % (i, j)
                tmp_data = list(self._read_lines(struct))
                tmp_arr = np.array(tmp_data)
                step, tmp_list = self._step_count(tmp_arr, self.num_trials, self.num_steps)
                step = int(step)
                for k in range(self.num_trials):
                    for l in range(self.num_steps):
                        self.all_data[cnt, 0:step, :] = tmp_list[k][l * step:(l + 1) * step]
                        self.data_labels_array[cnt, i] = 1.0
                        self.data_length_array[cnt] = step
                        cnt += 1
        self.data_steps_array = self.all_data[:, :, 4:16]  # to have last relevant data in csv files

    def _read_lines(self,file):
        with open(file, newline="") as data:
            reader = csv.reader(data)
            ind = 0
            for row in reader:
                if (ind > 0):  # not to include the first row
                    yield [float(i) for i in row]
                ind += 1

    def _step_count(self, raw_inp, num_trials, num_steps):
        cnt = 0
        inputs = [[] for i in range(num_trials)]
        for i in range(raw_inp.shape[0]):
            if i > 0:
                if (raw_inp[i, 3] != raw_inp[
                    i - 1, 3]):  # 3 is the column in csv files that shows the num of tiral
                    cnt += 1
            inputs[cnt].append(raw_inp[i])
        minimum = 1000000
        for i in range(num_trials):
            if (len(inputs[i]) < minimum):
                minimum = len(inputs[i])
        each_step = np.floor(minimum / num_steps)
        return each_step, inputs

if __name__ == "__main__":
    config_file = "./config/qcat.yaml"

    with open(config_file, 'r') as file_yaml:
        stink_config = yaml.safe_load(file_yaml)

    q_loader = QCAT_Loader(
        path_to_dataset=stink_config['path_to_dataset'],
        pick_modalities=stink_config['modalities'],
        save_to_pickle=stink_config['save_to_pickle']
    )
    print("finished")