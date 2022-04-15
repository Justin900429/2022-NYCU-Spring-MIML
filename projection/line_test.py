import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import ProjectionDataset
from model import model_define

plt.style.use("ggplot")


def plot_model(model, weight):
    # Set up the degree
    degree_list = list(range(10, 90, 10))
    plt.figure(figsize=(8, 6))

    model.load_weights(weight)

    for degree in degree_list:
        v = tf.random.uniform(shape=(1500,))
        norm_deg = np.array([degree / max_degree])
        norm_deg = norm_deg.repeat(v.shape[0])
        combine_features = np.stack([norm_deg, v], axis=1)

        res = model(combine_features)
        pred_height = res[:, 0].numpy()
        pred_range = res[:, 1].numpy()
        plt.scatter(
            (pred_range * max_x_range).tolist(),
            (pred_height * max_height).tolist(),
            s=5, label=rf"{degree}$^\circ$")

    plt.legend()
    plt.xlabel("R")
    plt.ylabel("H")
    plt.title(r"$R=4H\cot \theta$")
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

