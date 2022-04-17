import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import tensorflow as tf

from dataset import ProjectionDataset
from model import model_define

plt.style.use("ggplot")


def plot_model(model, weight):
    # Set up the degree
    degree_list = list(range(10, 90, 10))
    plt.figure(figsize=(8, 6))

    model.load_weights(weight)
    temp_m = 0

    for degree in degree_list:
        v = tf.random.uniform(shape=(1500,))
        norm_deg = np.array([degree / max_degree])
        norm_deg = norm_deg.repeat(v.shape[0])
        combine_features = np.stack([norm_deg, v], axis=1)

        res = model(combine_features)
        pred_height = res[:, 0].numpy()
        pred_range = res[:, 1].numpy()

        line_fit = LinearRegression().fit(
            pred_range * max_x_range,
            pred_height * max_height
        )
        m = line_fit.coef_.reshape(-1)[0]
        correct_coef = math.tan(math.radians(degree)) / 4
        cos_sim = compute_cosine_sim([1, m], [1, correct_coef])
        temp_m += cos_sim

        plt.scatter(
            (pred_range * max_x_range).tolist(),
            (pred_height * max_height).tolist(),
            s=5, label=rf"{degree}$^\circ$ m={cos_sim:.5f}")

    temp_m /= len(degree_list)
    plt.rcParams.update({
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0)
    })
    plt.legend()
    plt.xlabel("R")
    plt.ylabel("H")
    plt.title(rf"$R=4H\cot \theta$, sim={temp_m:.3f}")
    plt.grid(False)
    plt.savefig("test.pdf")
    plt.show()


if __name__ == "__main__":
    train_data = ProjectionDataset()

    max_degree = train_data.max_theta
    max_height = train_data.max_height
    max_x_range = train_data.max_x_range

    model = model_define()

    model.load_weights("model.h5")
    plot_model(model, "model.h5")

