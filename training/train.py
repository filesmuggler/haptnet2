import yaml
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

#from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedKFold

from loaders.HAPTNET_loader.HaptnetLoader import HAPTNET_Loader
from optimization.categorical_optimization import *

def train_model(dataset,config, num_classes):
    trainX, trainY, testX, testY = dataset
    batch_size = config['batch']



    for model_config in config['model_configs']:
        print("Running config: ", model_config['name'])
        modalities = model_config['modalities']
        num_modalities = len(modalities)
        fusion_type = model_config['fusion_type']

        if fusion_type == "None":
            if "force" in modalities:
                #TODO implement force case
                pass
            if "imu0" or "imu1" or "imu2" or "imu3" and not "force" in modalities:
                #TODO implement better condition
                pass
        elif fusion_type == "early":
            #TODO implement case
            pass
        elif fusion_type == "mid":
            # TODO implement case
            pass
        elif fusion_type == "late":
            # TODO implement case
            pass
        else:
            raise NotImplementedError("not implemented scenario")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "../logs/scalars/" + config['log_dir'] + model_config['name'] + "_" + model_config['fusion_type'] + "_" + timestamp

        # setup optimizer
        lr = float(train_config['learning_rate'])
        eta = tf.Variable(lr)
        eta_value = tf.keras.optimizers.schedules.ExponentialDecay(lr, 1000, 0.99)
        eta.assign(eta_value(0))
        optimizer = tf.keras.optimizers.Adam(eta)

        # setup writers
        logs_path = os.path.join(logdir)
        os.makedirs(logs_path, exist_ok=True)
        train_writer = tf.summary.create_file_writer(logs_path + "/train")
        val_writer = tf.summary.create_file_writer(logs_path + "/val")
        test_writer = tf.summary.create_file_writer(logs_path + "/test")

        #stratified sampling validation, validation dataset
        kf = StratifiedKFold(n_splits=int(config['num_folds']))
        for fold_no, (train_idx,val_idx) in enumerate(kf.split(X=trainX,y=trainY)):
            train_pick_x, train_pick_y = [trainX[i] for i in train_idx], trainY[train_idx]
            val_pick_x, val_pick_y = [trainX[i] for i in val_idx], trainY[val_idx]
            print("Processing kfold no ",fold_no)
            print(len(train_pick_y))
            print(len(val_pick_y))

            temp_train_X = []
            temp_val_X = []
            temp_test_X = []
            for i in range(num_modalities):
                temp_chunk = [modality[i] for modality in train_pick_x]
                temp_train_X.append(temp_chunk)
                temp_chunk = [modality[i] for modality in val_pick_x]
                temp_val_X.append(temp_chunk)
                temp_chunk = [modality[i] for modality in testX]
                temp_test_X.append(temp_chunk)

            train_x_reframed = tuple(temp_train_X)
            val_x_reframed = tuple(temp_val_X)
            test_x_reframed = tuple(temp_test_X)

            print("x")

            data_to_use_train = (train_x_reframed, train_pick_y)
            data_to_use_val = (val_x_reframed, val_pick_y)
            data_to_use_test = (test_x_reframed, testY)

            train_ds = tf.data.Dataset.from_tensor_slices(data_to_use_train). \
                shuffle(buffer_size=5000, reshuffle_each_iteration=True). \
                batch(batch_size=config['batch'])

            val_ds = tf.data.Dataset.from_tensor_slices(data_to_use_val). \
                shuffle(buffer_size=5000, reshuffle_each_iteration=True). \
                batch(batch_size=config['batch'])

            test_ds = tf.data.Dataset.from_tensor_slices(data_to_use_test).batch(config['batch'])

            train_step, val_step, test_step = 0, 0, 0

            num_epochs = config['num_epochs']

        print("x")

if __name__ == '__main__':
    config_file = "./config/train.yaml"

    with open(config_file, 'r') as file_yaml:
        train_config = yaml.safe_load(file_yaml)

    with open(train_config['loader_path'], 'r') as file_yaml:
        haptnet_config = yaml.safe_load(file_yaml)

    h_loader = HAPTNET_Loader(
        path_to_dataset=haptnet_config['path_to_dataset'],
        pick_modalities=haptnet_config['modalities'],
        save_to_pickle=haptnet_config['save_to_pickle']
    )

    dataset_train = h_loader.get_train_dataset()
    dataset_test = h_loader.get_test_dataset()

    x_train = [i[1:] for i in dataset_train]
    x_test = [i[1:] for i in dataset_test]

    y_train = [i[0] for i in dataset_train]
    y_test = [i[0] for i in dataset_test]

    unique, counts = np.unique(y_test, return_counts=True)

    num_classes = len(unique)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    dataset = [x_train, y_train, x_test, y_test]

    train_model(dataset,train_config,num_classes)

    print("x")
