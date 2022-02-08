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



def res_block(inputs_shape):
    x1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs_shape)
    x2 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x1)
    x1_residual = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs_shape)

    added = Add()([x1_residual, x2])
    return added

#input_shape = (1,3,1)



input_shape = (2,120,None)

# resnet blocks

inputs = Input(shape=input_shape)
inputs_f = Input(shape=(1,120,3))
inputs_q = Input(shape=(1,120,4))

resnet_block_1 = res_block(inputs_f)
resnet_block_2 = res_block(resnet_block_1)

resnet_block_3 = res_block(inputs_q)
resnet_block_4 = res_block(resnet_block_3)

res_add = Concatenate(axis=0)([resnet_block_4,resnet_block_2])
# lstm blocks

f_lstm_1 = LSTM(128,return_sequences=True, return_state=True)(res_add)
f_lstm_2 = LSTM(128,return_sequences=True, return_state=True)(f_lstm_1)
f_lstm_3 = LSTM(128,return_sequences=True, return_state=True)(f_lstm_2)

b_lstm_1 = LSTM(128,return_sequences=True,return_state=True,go_backwards=True)(res_add)
b_lstm_2 = LSTM(128,return_sequences=True,return_state=True,go_backwards=True)(b_lstm_1)
b_lstm_3 = LSTM(128,return_sequences=True,return_state=True,go_backwards=True)(b_lstm_2)

added_seq = Concatenate(axis=1)([f_lstm_3[0],b_lstm_3[0]])

added_flat = Flatten()(added_seq)
predictions_1 = Dense(100, activation='relu')(added_flat)
predictions_2 = Dense(6, activation='softmax')(predictions_1)
model = Model(inputs=inputs,outputs=predictions_2)
#model.add(DepthwiseConv2D(kernel_size=(1,3), activation='relu', input_shape=input_shape))
#model.add(SeparableConv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))


#model.add(LSTM(128, return_sequences=True, input_shape=(3,1)))
# input time steps
#data = np.array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
#print(model.predict(data))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#mock_data = np.random.rand(1,3)
mock_data = np.random.rand(120, 3)
mock_data_2 = np.random.rand(120, 4)

mock_data_t = tf.convert_to_tensor(mock_data)
mock_data_2_t = tf.convert_to_tensor(mock_data_2)
# mock_data_t = tf.expand_dims(mock_data_t, 0)
mock_data_t = tf.expand_dims(mock_data_t, 0)
mock_data_2_t = tf.expand_dims(mock_data_2_t, 0)

out = model(mock_data_t)
out_np = out.numpy()
print("X")