import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *

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
        conv_net = tf.keras.Sequential()
        for i, (num_filters, kernel, stride) in enumerate(zip(conv_filters,conv_kernels,conv_strides)):
            conv_net.add(tf.keras.layers.Conv1D(num_filters, kernel, stride, padding="SAME"))

            if i != len(self.config['conv_filters']) - 1:
                conv_net.add(tf.keras.layers.BatchNormalization())
                conv_net.add(tf.keras.layers.Activation("relu"))
                conv_net.add(tf.keras.layers.Dropout(dropout))

        return conv_net

    def _create_cnn_block(self, conv_types: list, conv_filters: list, conv_kernels: list, conv_strides: list, dropout: float, input_shape):
        self.conv_layers = []
        for i, (type, num_filters, kernel, stride) in enumerate(zip(conv_types, conv_filters, conv_kernels, conv_strides)):
            c = Conv1D(filters=num_filters, kernel_size=kernel, stride=stride, dropout=dropout, padding="SAME")


    def _add_conv_layer(self, filter_size: int, kernel_size: int, stride: int, dropout: float):
        c = Conv1D(filters=filter_size, kernel_size=kernel_size, stride=stride, dropout=dropout, padding="SAME")


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