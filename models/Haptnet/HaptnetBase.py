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

    def call(self, inputs, training=None):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _add_fc_block(self, fc_layers: list, dropout: float):
        fc_net = tf.keras.Sequential()
        # fc_net.add(tf.keras.layers.Flatten())
        for i, fc_units in enumerate(fc_layers):
            fc_net.add(tf.keras.layers.Dense(fc_units))
            if i != len(fc_layers) - 1:
                fc_net.add(tf.keras.layers.BatchNormalization())
                fc_net.add(tf.keras.layers.Activation("relu"))
                fc_net.add(tf.keras.layers.Dropout(dropout))

        return fc_net

    def _create_fc_block(self, fc_layers: list, dropout: float):
        fc_net = []
        for i, fc_units in enumerate(fc_layers):
            if i == len(fc_layers)-1:
                d = Dense_Block(neurons=fc_units, dropout=dropout,last=True)
            else:
                d = Dense_Block(neurons=fc_units, dropout=dropout, last=False)
            fc_net.append(d)

        return fc_net

    def _create_cnn_block(self, conv_types: list, conv_filters: list, conv_kernels: list, conv_strides: list,
                          dropout: float,
                          input_shape: tuple):
        conv_net = []
        for i, (conv_type, num_filters, kernel, stride) in enumerate(
                zip(conv_types, conv_filters, conv_kernels, conv_strides)):
            if conv_type == "None":
                if i == 0:
                    c = CNN_1D_Block(out_channels=num_filters, kernel_size=kernel, stride=stride,
                                     dropout=dropout).model(input_shape)
                else:
                    c = CNN_1D_Block(out_channels=num_filters, kernel_size=kernel, stride=stride, dropout=dropout)
            elif conv_type == "SingleHop":
                if i == 0:
                    c = Res_CNN1D_S_Block(channels=num_filters, kernel_size=kernel, stride=stride,
                                          dropout=dropout).model(input_shape)
                else:
                    c = Res_CNN1D_S_Block(channels=num_filters, kernel_size=kernel, stride=stride, dropout=dropout)
            elif conv_type == "DoubleHop":
                if i == 0:
                    c = Res_CNN1D_D_Block(channels=[num_filters, num_filters],
                                          kernel_size=[kernel, kernel],
                                          stride=[stride, stride],
                                          dropout=dropout).model(input_shape)
                elif i == len(conv_filters)-1:
                    c = Res_CNN1D_D_Block(channels=[num_filters, num_filters],
                                          kernel_size=[kernel, kernel],
                                          stride=[stride, stride],
                                          dropout=dropout)
                else:
                    c = Res_CNN1D_D_Block(channels=[num_filters, num_filters],
                                          kernel_size=[kernel, kernel],
                                          stride=[stride, stride],
                                          dropout=dropout)
            else:
                raise NotImplementedError

            conv_net.append(c)

        return conv_net

    def _create_lstm_block(self, lstm_units: int, return_sequences: bool, lstm_nest: int, dropout: float,
                           stateful: bool):

        forward_model = []
        backward_model = []

        for i in range(lstm_nest):
            f_rnn = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, dropout=dropout,
                                         stateful=stateful, dtype=tf.float32)
            b_rnn = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, go_backwards=True,
                                         dropout=dropout, stateful=stateful, dtype=tf.float32)
            forward_model.append(f_rnn)
            backward_model.append(b_rnn)

        return forward_model, backward_model
