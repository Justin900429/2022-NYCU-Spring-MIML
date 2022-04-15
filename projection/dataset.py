import math
import random
import numpy as np
import tensorflow as tf


class ProjectionDataset(tf.keras.utils.Sequence):
    # Constant for the gravity
    g = 9.81

    def __init__(self, max_theta=90, max_velocity=100, batch_size=4096):
        self.max_theta = max_theta
        self.max_velocity = max_velocity
        self.max_height = (max_velocity**2) / (2 * self.g)
        self.max_x_range = (max_velocity**2) / self.g
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __len__(self):
        return 1

    def __generate_data(self):
        # You don't need idx in fact, just randomly generate the data
        theta = random.random() * self.max_theta
        velocity = random.random() * self.max_velocity

        # Compute the real answer
        height = (velocity * math.sin(math.radians(theta)))**2 / (2 * self.g)
        x_range = (velocity**2) * math.sin(math.radians(theta * 2)) / self.g
        data = np.array([
            height / self.max_height,
            x_range / self.max_x_range
        ])
        combine = np.array([
            theta / self.max_theta,
            velocity / self.max_velocity
        ])

        return data, combine

    def __generate_batch(self):
        data_list, combine_list = [], []

        for _ in range(self.batch_size):
            data, combine = self.__generate_data()
            data_list.append(data)
            combine_list.append(combine)

        return np.array(combine_list), np.array(data_list)

    def __getitem__(self, idx):
        return self.__generate_batch()

