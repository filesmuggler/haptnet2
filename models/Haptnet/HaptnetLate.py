import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow import keras
from tensorflow.keras.layers import *
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

        temp_outs = []

        for i,(modality,input_tensor) in enumerate(zip(self.modalities,inputs)):

            x = input_tensor
            for c_block in self.cnn_body[i]:
                x = c_block(x, training=training)

            x_f_rnn, x_b_rnn = x, x


            for f_r_block in self.f_rnn_body[i]:
                x_f_rnn = f_r_block(x_f_rnn, training=training)

            for b_r_block in self.b_rnn_body[i]:
                x_b_rnn = b_r_block(x_b_rnn, training=training)

            x_conc = concatenate([x_f_rnn,x_b_rnn])
            x_fc = x_conc

            x_fc = Flatten()(x_fc)


            for fc_block in self.fc_body[i]:
                x_fc = fc_block(x_fc,training=training)

            temp_outs.append(x_fc)

        if len(temp_outs)>1:
            x_out = Add()(temp_outs)
        else:
            x_out = temp_outs[0]

        return x_out

    def _create_model(self):
        num_mod = len(self.modalities)
        self.cnn_body, self.f_rnn_body, self.b_rnn_body, self.fc_body = [],[],[],[]

        for m in self.modalities:
            if m=="force":
                input_shape = (400,3)
            elif "imu" in m:
                input_shape = (400,4)
            else:
                raise NotImplementedError

            cnn_head = self._create_cnn_block(self.config['conv_types'],
                                          self.config['conv_filters'],
                                          self.config['conv_kernels'],
                                          self.config['conv_strides'],
                                          self.config['dropout'],
                                          input_shape)
            self.cnn_body.append(cnn_head)

            f_rnn_head, b_rnn_head = self._create_lstm_block(self.config['lstm_units'],
                                               self.config['return_sequences'],
                                               self.config['lstm_nest'],
                                               self.config['dropout'],
                                               self.config['stateful'])
            self.f_rnn_body.append(f_rnn_head)
            self.b_rnn_body.append(b_rnn_head)

            fc_head = self._create_fc_block(self.config['fc_units'],self.config['dropout'])

            self.fc_body.append(fc_head)

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))





