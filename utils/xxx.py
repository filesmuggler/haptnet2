import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
import numpy as np

# model = Sequential()
# model.add(SeparableConv1D(filters=64, kernel_size=3, activation='relu', input_shape=(400,3)))
# model.add(SeparableConv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(6, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()


model = Sequential()
input_shape = (1,120,3)
#input_shape = (120,3)
#model.add(Conv2D(64, kernel_size=(3,1), activation='relu', input_shape=input_shape))
model.add(DepthwiseConv2D(kernel_size=(1,3), activation='relu', input_shape=input_shape))
#model.add(SeparableConv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

mock_data = np.random.rand(1,120,3)
#mock_data = np.random.rand(120, 3)

mock_data_t = tf.convert_to_tensor(mock_data)
mock_data_t = tf.expand_dims(mock_data_t, 0)

out = model(mock_data_t)
out_np = out.numpy()
print("X")