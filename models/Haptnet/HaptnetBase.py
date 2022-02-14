import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from utils.HaptUtils.hapt_blocks import *

class HaptnetBase(tf.keras.Model):
    def __init__(self, batch_size: int,
                 num_outputs: int,
                 config: dict,
                 modalities: list,
                 fusion_type: str,
                 *args, **kwargs):
        super(HaptnetBase, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_outputs = num_outputs
        self.config = config
        self.modalities = modalities
        self.fusion_type = fusion_type
        self._create_model()

    def call(self,inputs,training=None):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _add_fc_block(self,fc_layers: list, dropout: float):
        fc_net = tf.keras.Sequential()
        #fc_net.add(tf.keras.layers.Flatten())
        for i, fc_units in enumerate(fc_layers):
            fc_net.add(tf.keras.layers.Dense(fc_units))
            if i != len(fc_layers) - 1:
                fc_net.add(tf.keras.layers.BatchNormalization())
                fc_net.add(tf.keras.layers.Activation("relu"))
                fc_net.add(tf.keras.layers.Dropout(dropout))

        return fc_net

    def _add_conv_block(self, conv_filters: list, conv_kernels: list, conv_strides: list, dropout: float):
        # DEPRECATED
        conv_net = tf.keras.Sequential()
        for i, (num_filters, kernel, stride) in enumerate(zip(conv_filters,conv_kernels,conv_strides)):
            conv_net.add(tf.keras.layers.Conv1D(num_filters, kernel, stride, padding="SAME"))

            if i != len(self.config['conv_filters']) - 1:
                conv_net.add(tf.keras.layers.BatchNormalization())
                conv_net.add(tf.keras.layers.Activation("relu"))
                conv_net.add(tf.keras.layers.Dropout(dropout))

        return conv_net

    def _create_cnn_block(self, conv_types: list, conv_filters: list, conv_kernels: list, conv_strides: list, dropout: float):
        conv_net = []
        for i, (conv_type, num_filters, kernel, stride) in enumerate(zip(conv_types, conv_filters, conv_kernels, conv_strides)):
            if conv_type == "None":
                c = CNN_1D_Block(out_channels=num_filters,kernel_size=kernel,stride=stride,dropout=dropout)
                conv_net.append(c)
            elif conv_type == "SingleHop":
                c = Res_CNN1D_S_Block(channels=num_filters,kernel_size=kernel,stride=stride,dropout=dropout)
                conv_net.append(c)
            elif conv_type == "DoubleHop":
                c = Res_CNN1D_D_Block(channels=[num_filters,num_filters],
                                      kernel_size=[kernel,kernel],
                                      stride=[stride,stride],
                                      dropout=dropout)
                conv_net.append(c)
            else:
                raise NotImplementedError

        return conv_net


    def _add_lstm_block(self, lstm_units: int, return_seq: bool, dropout: float, stateful: bool):
        fwd_block = tf.keras.Sequential()

        fwd_layer = tf.keras.layers.LSTM(lstm_units,
                                         return_sequences=return_seq,
                                         dropout=dropout,
                                         stateful=stateful,
                                         dtype=tf.float32)

        bckwd_layer = tf.keras.layers.LSTM(lstm_units,
                                         return_sequences=return_seq,
                                         go_backwards=True,
                                         dropout=dropout,
                                         stateful=stateful,
                                         dtype=tf.float32)

        aggregator = tf.keras.layers.Bidirectional(fwd_layer,
                                                   backward_layer=bckwd_layer,
                                                   input_shape=(self.batch_size,int(2 * lstm_units))
                                                   )

        return aggregator


    def _create_lstm_block(self, lstm_units: int, return_sequences: bool, lstm_nest: int):

        forward_model = keras.Model()

        for i in range(lstm_nest):
