import yaml
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedKFold

from loaders.HAPTNET_loader.HaptnetLoader import HAPTNET_Loader
from optimization.categorical_optimization import *
from models.Haptnet.HaptnetLate import HaptnetLate

def train_model(dataset,config, num_classes):
    trainX, trainY, testX, testY = dataset
    batch_size = config['batch']
    print("dupa")
    for model_config in config['model_configs']:
        print("Running config: ", model_config['name'])
        modalities = model_config['modalities']
        num_modalities = len(modalities)
        fusion_type = model_config['fusion_type']

        if fusion_type == "None":
            raise NotImplementedError("not implemented scenario")
        elif fusion_type == "early":
            #TODO implement case
            raise NotImplementedError("not implemented scenario")
        elif fusion_type == "mid":
            # TODO implement case
            raise NotImplementedError("not implemented scenario")
        elif fusion_type == "late":

            model = HaptnetLate(batch_size, 6, model_config, model_config['modalities'], model_config['fusion_type'])

            print("haha")
        else:
            raise NotImplementedError("not implemented scenario")



        # setup optimizer
        lr = float(train_config['learning_rate'])
        eta = tf.Variable(lr)
        eta_value = tf.keras.optimizers.schedules.ExponentialDecay(lr, 1000, 0.99)
        eta.assign(eta_value(0))
        optimizer = tf.keras.optimizers.Adam(eta)

        #stratified sampling validation, validation dataset
        kf = IterativeStratification(n_splits=int(config['num_folds']), order=1)

        for fold_no, (train_idx,val_idx) in enumerate(kf.split(X=trainX,y=trainY)):

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = "../logs/scalars/" + config['log_dir'] + model_config['name'] + "_" + model_config[
                'fusion_type'] + "_fold_" + str(fold_no) + "_" + timestamp
            print(logdir)
            # setup writers
            logs_path = os.path.join(logdir)
            os.makedirs(logs_path, exist_ok=True)
            train_writer = tf.summary.create_file_writer(logs_path + "/train")
            val_writer = tf.summary.create_file_writer(logs_path + "/val")
            test_writer = tf.summary.create_file_writer(logs_path + "/test")

            train_pick_x, train_pick_y = [trainX[i] for i in train_idx], trainY[train_idx]
            val_pick_x, val_pick_y = [trainX[i] for i in val_idx], trainY[val_idx]
            print("Processing kfold no ",fold_no)
            print(len(train_pick_y))
            print(len(val_pick_y))

            temp_train_X = []
            temp_val_X = []
            temp_test_X = []

            for m in modalities:
                if m=="force":
                    temp_chunk = [data_packet[0] for data_packet in train_pick_x]
                    temp_train_X.append(temp_chunk)
                    temp_chunk = [data_packet[0] for data_packet in val_pick_x]
                    temp_val_X.append(temp_chunk)
                    temp_chunk = [data_packet[0] for data_packet in testX]
                    temp_test_X.append(temp_chunk)
                if m=="imu0":
                    temp_chunk = [data_packet[1] for data_packet in train_pick_x]
                    temp_train_X.append(temp_chunk)
                    temp_chunk = [data_packet[1] for data_packet in val_pick_x]
                    temp_val_X.append(temp_chunk)
                    temp_chunk = [data_packet[1] for data_packet in testX]
                    temp_test_X.append(temp_chunk)
                if m=="imu1":
                    temp_chunk = [data_packet[2] for data_packet in train_pick_x]
                    temp_train_X.append(temp_chunk)
                    temp_chunk = [data_packet[2] for data_packet in val_pick_x]
                    temp_val_X.append(temp_chunk)
                    temp_chunk = [data_packet[2] for data_packet in testX]
                    temp_test_X.append(temp_chunk)
                if m=="imu2":
                    temp_chunk = [data_packet[3] for data_packet in train_pick_x]
                    temp_train_X.append(temp_chunk)
                    temp_chunk = [data_packet[3] for data_packet in val_pick_x]
                    temp_val_X.append(temp_chunk)
                    temp_chunk = [data_packet[3] for data_packet in testX]
                    temp_test_X.append(temp_chunk)
                if m=="imu3":
                    temp_chunk = [data_packet[4] for data_packet in train_pick_x]
                    temp_train_X.append(temp_chunk)
                    temp_chunk = [data_packet[4] for data_packet in val_pick_x]
                    temp_val_X.append(temp_chunk)
                    temp_chunk = [data_packet[4] for data_packet in testX]
                    temp_test_X.append(temp_chunk)

            # for i in range(num_modalities):
            #     temp_chunk = [modality[i] for modality in train_pick_x]
            #     temp_train_X.append(temp_chunk)
            #     temp_chunk = [modality[i] for modality in val_pick_x]
            #     temp_val_X.append(temp_chunk)
            #     temp_chunk = [modality[i] for modality in testX]
            #     temp_test_X.append(temp_chunk)

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
            for epoch in range(num_epochs):
                print(f'running {epoch} epoch out of {num_epochs} of {fold_no} fold')
                train_step = train_classification(model, train_writer, train_ds, optimizer, train_step, num_classes)
                val_step = validate_classification(model, val_writer, val_ds, val_step, num_classes, "val")
                test_step = validate_classification(model, test_writer, test_ds, test_step, num_classes, "test")
            model_name = model_config['name'] + "_" + model_config['fusion_type'] + "_fold_" + str(fold_no) + "_" + timestamp
            model.save("../saved_models/models/" + model_name)


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
    y_train = y_train[:,0,:]
    y_test = np.array(y_test)
    y_test = y_test[:, 0, :]

    dataset = [x_train, y_train, x_test, y_test]

    train_model(dataset,train_config,num_classes)

    print("x")
