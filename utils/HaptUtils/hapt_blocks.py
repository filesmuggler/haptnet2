import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNN_1D_Block(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding="same", activation="relu", dropout=0.5, beta=1.0):
        super(CNN_1D_Block, self).__init__()
        self.conv = layers.Conv1D(out_channels, kernel_size, stride=stride, padding=padding)
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
        else:
            return NotImplementedError
        return x


class CNN_2D_Block(layers.Layer):
    def __init__(self, out_channels: int,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0):
        super(CNN_2D_Block, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, stride=stride, padding=padding)
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

class Res_CNN1D_S_Block(layers.Layer):
    def __init__(self, channels: int,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0):
        super(Res_CNN1D_S_Block, self).__init__()

        self.cnn = CNN_1D_Block(out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                activation=activation,
                                dropout=dropout,
                                beta=beta)

        self.identity_mapping = CNN_1D_Block(
                                out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride[1],
                                padding="same",
                                activation="None",
                                dropout=0.0,
                                beta=beta)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.cnn(input_tensor,training=training)
        x = self.identity_mapping(x,training=training)
        x = layers.Add([input_tensor, x])
        return x

class Res_CNN1D_D_Block(layers.Layer):
    def __init__(self, channels: list,
                 kernel_size: list,
                 stride: list,
                 padding="same",
                 activation="relu",
                 dropout=0.5,
                 beta=1.0):
        super(Res_CNN1D_D_Block, self).__init__()

        self.cnn1 = CNN_1D_Block(out_channels=channels[0],
                                kernel_size=kernel_size[0],
                                stride=stride[0],
                                padding=padding,
                                activation=activation,
                                dropout=dropout,
                                beta=beta)

        self.cnn2 = CNN_1D_Block(out_channels=channels[1],
                                 kernel_size=kernel_size[1],
                                 stride=stride[1],
                                 padding=padding,
                                 activation=activation,
                                 dropout=dropout,
                                 beta=beta)

        self.identity_mapping = CNN_1D_Block(
                                out_channels=channels[1],
                                kernel_size=kernel_size[1],
                                stride=stride[1],
                                padding="same",
                                activation="None",
                                dropout=0.0,
                                beta=beta)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.cnn1(input_tensor,training=training)
        x = self.cnn2(x,training=training)
        x = self.identity_mapping(x,training=training)
        x = layers.Add([input_tensor, x])
        return x

