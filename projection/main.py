import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tqdm import tqdm

from dataset import ProjectionDataset
from model import model_define


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, weights_file, monitor="loss", mode="min",
                 save_weights_only=False):
        super().__init__()

        self.weights_file = weights_file
        self.monitor = monitor

        self.mode = mode
        self.wave_weights_only = save_weights_only

        if mode == "min":
            self.best = np.Inf
        else:
            self.best = -np.Inf

    def save_model(self):
        if self.wave_weights_only:
            self.model.save_weights(self.weights_file)
        else:
            self.model.save(self.weights_file)

    def on_epoch_end(self, epoch, logs=None):
        monitor_value = logs.get(self.monitor)

        if self.mode == "min" and monitor_value < self.best:
            self.save_model()
            self.best = monitor_value
        elif self.mode == "max" and monitor_value > self.best:
            self.save_model()
            self.best = monitor_value


def mse_loss(y_true, y_pred):
    diff = tf.reduce_sum((y_true - y_pred) ** 2, axis=-1)
    return tf.reduce_mean(diff)


def scheduler_func(epoch, lr):
    if (epoch != 0) and (epoch % 500 == 0):
        return lr * 0.7
    else:
        return lr


class Progressbar(tf.keras.callbacks.Callback):
    def __init__(self, epochs, monitor="loss"):
        super().__init__()
        self.monitor = monitor
        self.epochs = epochs
        self.pbar = tqdm(total=epochs)

    def on_epoch_end(self, epoch, logs=None):
        monitor_value = logs.get(self.monitor)
        self.pbar.update(epoch)

        if epoch == self.epochs - 1:
            self.pbar.close()


if __name__ == "__main__":
    print(device_lib.list_local_devices())

    # Create model and dataset
    model = model_define()
    model.compile(keras.optimizers.Adam(learning_rate=5e-2),
                  loss=tf.keras.losses.MeanSquaredError())

    train_data = ProjectionDataset()

    model.fit(train_data, epochs=10000,
              callbacks=[
                SaveModel("model.h5"),
                tf.keras.callbacks.LearningRateScheduler(scheduler_func),
              ], verbose=0)

