import math
import random

import torch
from torch.utils.data import Dataset, DataLoader


class Projection(Dataset):
    # Constant for the gravity
    g = 9.81

    def __init__(self, max_theta, max_velocity, batch_size):
        self.max_theta = max_theta
        self.max_velocity = max_velocity
        self.max_height = (max_velocity**2) / (2 * self.g)
        self.max_x_range = (max_velocity**2) / self.g
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        # You don't need idx in fact, just randomly generate the data
        theta = random.random() * self.max_theta
        velocity = random.random() * self.max_velocity

        # Compute the real answer
        height = (velocity * math.sin(math.radians(theta)))**2 / (2 * self.g)
        x_range = (velocity**2) * math.sin(math.radians(theta * 2)) / self.g
        height = torch.tensor([height / self.max_height])
        x_range = torch.tensor([x_range / self.max_x_range])

        combine = torch.cat([
            torch.tensor([theta / self.max_theta]),
            torch.tensor([velocity / self.max_velocity])
        ])

        return combine, (height, x_range)


def make_loader(max_theta=90.0, max_velocity=100, batch_size=128):
    dataset = Projection(max_theta=max_theta, max_velocity=max_velocity, batch_size=batch_size)
    return DataLoader(dataset, batch_size=batch_size)
