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

        for m in self.modalities:
            if m == "force":
                self.force_head = self._create_cnn_block(
                                            self.config['conv_types'],
                                            self.config['conv_filters'],
                                            self.config['conv_kernels'],
                                            self.config['conv_strides'],
                                            self.config['dropout'])
            if m == "imu0":
                self.imu0_head = self._create_cnn_block(
                                            self.config['conv_types'],
                                            self.config['conv_filters'],
                                            self.config['conv_kernels'],
                                            self.config['conv_strides'],
                                            self.config['dropout'])
            if m == "imu1":
                self.imu1_head = self._create_cnn_block(
                                            self.config['conv_types'],
                                            self.config['conv_filters'],
                                            self.config['conv_kernels'],
                                            self.config['conv_strides'],
                                            self.config['dropout'])
            if m == "imu2":
                self.imu2_head = self._create_cnn_block(
                                            self.config['conv_types'],
                                            self.config['conv_filters'],
                                            self.config['conv_kernels'],
                                            self.config['conv_strides'],
                                            self.config['dropout'])
            if m == "imu3":
                self.imu3_head = self._create_cnn_block(
                                            self.config['conv_types'],
                                            self.config['conv_filters'],
                                            self.config['conv_kernels'],
                                            self.config['conv_strides'],
                                            self.config['dropout'])
            print("hihi")






