import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class HaptnetLate(tf.keras.Model):
    def __init__(self, batch_size: int,
                 num_outputs: int,
                 config: dict,
                 modalities: list,
                 fusion_type: str,
                 *args, **kwargs):
        super(HaptnetLate, self).__init__(*args, **kwargs)
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

        self._create_cnn_block(self.config,inputs_shape)


