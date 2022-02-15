import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
import numpy as np

from HaptUtils.hapt_blocks import *


class ResNetLike(keras.Model):
    def __init__(self, name="None"):
        super(ResNetLike, self).__init__()
        self.b_1 = Res_CNN1D_D_Block(channels=[3,3],kernel_size=[3,3],stride=[1,1])
        self.b_2 = Res_CNN1D_D_Block(channels=[3,3],kernel_size=[3,3],stride=[1,1])

    def call(self, input_tensor, training=False):
        x = self.b_1(input_tensor, training=training)
        x = self.b_2(x, training=training)
        return x

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

input_shape_forces = (120,3)
model_f = ResNetLike().model(input_shape_forces)

input_shape_quats = (120,16)
model_q = ResNetLike().model(input_shape_quats)

x = concatenate([model_f.layers[-1].output,model_q.layers[-1].output])
x = Flatten()(x)
x = Dense(100,activation="relu")(x)
x = Dense(6,activation="relu")(x)

force_input = model_f.layers[0].input
quats_input = model_q.layers[0].input
#output = layers.Dense(10)(layers.Flatten()(base_output))
model = keras.Model([force_input,quats_input], x)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


mock_data_f = np.random.rand(120, 3)
mock_data_q = np.random.rand(120, 16)

mock_data_ft = tf.convert_to_tensor(mock_data_f)
mock_data_ft = tf.expand_dims(mock_data_ft, 0)
#mock_data_ft = tf.expand_dims(mock_data_ft, 0)

mock_data_qt = tf.convert_to_tensor(mock_data_q)
mock_data_qt = tf.expand_dims(mock_data_qt, 0)
#mock_data_qt = tf.expand_dims(mock_data_qt, 0)

in_data = ()

out = model([mock_data_ft,mock_data_qt])
out_np = out.numpy()
print("dupa")