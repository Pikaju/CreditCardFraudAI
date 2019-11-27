import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FraudDetectionMLP(keras.Model):
    def __init__(self):
        super(FraudDetectionMLP, self).__init__()
        self.dense1 = layers.Dense(16, activation=tf.nn.relu)
        self.dense2 = layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        x = inputs
        x = self.dense1(x)
        # x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x
