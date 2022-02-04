import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from HaptnetBase import HaptnetBase

class HaptnetForce(HaptnetBase):
    def __init__(self, batch_size: int,
                 num_outputs: int,
                 config: dict,
                 modalities: list,
                 fusion_type: str,
                 *args, **kwargs):
        super(HaptnetForce, self).__init__(batch_size, num_outputs, config, modalities,
                 fusion_type, *args, **kwargs)
        self._create_model()


    def _create_model(self):
        conv_block = self._add_conv_block(self.config['conv_filters'],
                                          self.config['conv_kernels'],
                                          self.config['conv_strides'],
                                          self.config['dropout'])

        fc_block = self._add_fc_block(self.config['fc_units'],
                                      self.config['dropout'])

