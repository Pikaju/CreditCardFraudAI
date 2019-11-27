import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FraudDetectionMLP(keras.Model):
    def __init__(self):
        super(FraudDetectionMLP, self).__init__()
        self.dense3 = layers.Dense(64, activation=tf.nn.relu)
        self.dense4 = layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        x = inputs
        x = self.dense3(x)
        x = self.dropout(x, training=training)
        x = self.dense4(x)
        return x
