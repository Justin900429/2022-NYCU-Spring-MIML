from tensorflow import keras
from tensorflow.keras import layers


def model_define():
    inputs = keras.Input(shape=(2))
    x = layers.Dense(16, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)

