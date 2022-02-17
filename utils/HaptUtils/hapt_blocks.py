import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Dense_Block(layers.Layer):
    def __init__(self, neurons: int,
                 activation="relu",
                 dropout=0.5,
                 beta=1.0,
                 name="dense",
                 last=False):
        super(Dense_Block, self).__init__()
        self.dense = layers.Dense(neurons,name=name)
        self.drop = layers.Dropout(rate=dropout)
        self.bn = layers.BatchNormalization()
        self.activation = activation
        self.beta = beta
        self.dropout = dropout
        self.last = last

    def call(self, input_tensor, training=False, **kwargs):
        x = self.dense(input_tensor)
        if not self.last:
            x = self.bn(x, training=training)
            if self.activation == "relu":
                x = tf.nn.relu(x)
            elif self.activation == "silu":
                x = tf.nn.silu(x, beta=self.beta)
            elif self.activation == "gelu":
                x = tf.nn.gelu(x)
            elif self.activation == "None":
                x = self.drop(x, training=training)
                return x
            else:
                return NotImplementedError
            x = self.drop(x, training=training)
        return x

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

class CNN_1D_Block(layers.Layer):
    def __init__(self, out_channels: int,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0,
                 name="conv"):
        super(CNN_1D_Block, self).__init__()
        self.conv = layers.Conv1D(out_channels, kernel_size, strides=stride, padding=padding, name=name)
        self.bn = layers.BatchNormalization()
        self.activation = activation
        self.beta = beta
        self.dropout = dropout

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        if self.activation == "relu":
            x = tf.nn.relu(x)
        elif self.activation == "silu":
            x = tf.nn.silu(x, beta=self.beta)
        elif self.activation == "gelu":
            x = tf.nn.gelu(x)
        elif self.activation == "None":
            return x
        else:
            return NotImplementedError
        return x

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

class CNN_2D_Block(layers.Layer):
    def __init__(self, out_channels: int,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0,
                 name="conv"):
        super(CNN_2D_Block, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, strides=stride, padding=padding, name=name)
        self.bn = layers.BatchNormalization()
        self.activation = activation
        self.beta = beta
        self.dropout = dropout

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        if self.activation == "relu":
            x = tf.nn.relu(x)
        elif self.activation == "silu":
            x = tf.nn.silu(x, beta=self.beta)
        elif self.activation == "gelu":
            x = tf.nn.gelu(x)
        elif self.activation == "None":
            return x
        else:
            return NotImplementedError
        return x

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

class Res_CNN1D_S_Block(layers.Layer):
    def __init__(self, channels: int,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0,
                 name="res_s"):
        super(Res_CNN1D_S_Block, self).__init__()

        self.cnn = CNN_1D_Block(out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                activation=activation,
                                dropout=dropout,
                                beta=beta,
                                name=name+"_conv_1")

        self.identity_mapping = CNN_1D_Block(
                                out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                activation="None",
                                dropout=0.0,
                                beta=beta,
                                name=name+"_iden")

    def call(self, input_tensor, training=False, **kwargs):
        x = self.cnn(input_tensor,training=training)
        iden = self.identity_mapping(input_tensor,training=training)
        x = layers.Add()([iden, x])
        return x

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

class Res_CNN1D_D_Block(layers.Layer):
    def __init__(self, channels: list,
                 kernel_size: list,
                 stride: list,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0,
                 name="res_d"):
        super(Res_CNN1D_D_Block, self).__init__()

        self.cnn1 = CNN_1D_Block(out_channels=channels[0],
                                kernel_size=kernel_size[0],
                                stride=stride[0],
                                padding=padding,
                                activation=activation,
                                dropout=dropout,
                                beta=beta,
                                name=name+"_conv_1")

        self.cnn2 = CNN_1D_Block(out_channels=channels[1],
                                 kernel_size=kernel_size[1],
                                 stride=stride[1],
                                 padding=padding,
                                 activation=activation,
                                 dropout=dropout,
                                 beta=beta,
                                 name=name+"_conv_2")

        self.identity_mapping = CNN_1D_Block(
                                out_channels=channels[1],
                                kernel_size=2*kernel_size[1],
                                stride=2*stride[1],
                                padding="same",
                                activation="None",
                                dropout=0.0,
                                beta=beta,
                                name=name+"_iden")

    def call(self, input_tensor, training=False, **kwargs):
        x = self.cnn1(input_tensor,training=training)
        x = self.cnn2(x,training=training)
        iden = self.identity_mapping(input_tensor,training=training)
        x = layers.Add()([iden, x])
        return x

    def model(self,inputs_shape):
        x = keras.Input(shape=inputs_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


