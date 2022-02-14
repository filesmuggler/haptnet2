import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.HaptUtils.hapt_blocks import *
from models.Haptnet.HaptnetBase import HaptnetBase

class HaptnetLate(HaptnetBase):
    def __init__(self, batch_size: int,
                 num_outputs: int,
                 config: dict,
                 modalities: list,
                 fusion_type: str,
                 *args, **kwargs):
        super(HaptnetLate, self).__init__(
                 batch_size,
                 num_outputs,
                 config,
                 modalities,
                 fusion_type,
                 *args, **kwargs)
        self.batch_size = batch_size
        self.num_outputs = num_outputs
        self.config = config
        self.modalities = modalities
        self.fusion_type = fusion_type
        self._create_model()

    def call(self,inputs,training=None):
        for input_modality in inputs:
            print("shape: ",input_modality.shape)

    def _create_model(self):
        num_mod = len(self.modalities)
        self.heads = []
        for m in self.modalities:
            head = self._create_cnn_block(self.config['conv_types'],
                                          self.config['conv_filters'],
                                          self.config['conv_kernels'],
                                          self.config['conv_strides'],
                                          self.config['dropout'])

            self.heads.append(head)

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))





