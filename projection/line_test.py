import argparse

import math
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch

from model import ProjectileModel
from dataset import make_loader

plt.style.use("ggplot")


def compute_cosine_sim(x, y):
    if isinstance(x, (list, tuple)):
        x = np.array(x)

    if isinstance(y, (list, tuple)):
        y = np.array(y)

    return (x / np.linalg.norm(x)) @ (y / np.linalg.norm(y)).T


@torch.no_grad()
def line_test(args):
    if "cuda" in args.device:
        args.cuda = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Create model
    model = ProjectileModel(args.in_features, args.hidden_features, args.out_features).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    # Create dataloader
    dataloader = make_loader(args.max_t, args.max_v, args.batch_size)

    dataset_max_degree = dataloader.dataset.max_theta
    dataset_max_height = dataloader.dataset.max_height
    dataset_max_x_range = dataloader.dataset.max_x_range

    # Set up the degree
    degree_list = list(range(10, 90, 10))
    plt.figure(figsize=(8, 6))

    temp_m = 0

    for degree in degree_list:
        v = torch.rand(1500)
        norm_deg = torch.tensor([degree / dataset_max_degree])
        norm_deg = norm_deg.expand(v.shape[0])
        combine_features = torch.stack([norm_deg, v], dim=-1).to(device)

        pred_height, pred_range = model(combine_features)

        line_fit = LinearRegression().fit(
            (pred_range * dataset_max_x_range).numpy(), (pred_height * dataset_max_height).numpy()
        )
        m = line_fit.coef_.reshape(-1)[0]

        correct_coef = (1 / math.tan(degree))
        cos_sim = compute_cosine_sim([1, m], [1, correct_coef])
        temp_m += cos_sim

        plt.scatter(
            (pred_range * dataset_max_x_range).tolist(),
            (pred_height * dataset_max_height).tolist(),
            s=5, label=rf"{degree}$^\circ$, m={cos_sim:.3f}")

    temp_m /= len(degree_list)
    plt.legend()
    plt.xlabel("R")
    plt.ylabel("H")
    plt.title(rf"$R=4H\cot \theta$, sim={temp_m:.3f}")
    plt.grid(False)
    plt.savefig(args.outfile)
    plt.show()


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--out_features", type=int, default=1)
    parser.add_argument("--weight", type=str, default="model.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_t", type=float, default=90)
    parser.add_argument("--max_v", type=float, default=100)
    parser.add_argument("--outfile", type=str, default="plot/line_test.pdf")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    line_test(args)
