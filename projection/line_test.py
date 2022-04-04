import argparse
import torch
import matplotlib.pyplot as plt

from model import ProjectileModel
from dataset import make_loader

plt.style.use("ggplot")


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

    for degree in degree_list:
        v = torch.rand(1500)
        norm_deg = torch.tensor([degree / dataset_max_degree])
        norm_deg = norm_deg.expand(v.shape[0])
        combine_features = torch.stack([norm_deg, v], dim=-1).to(device)

        pred_height, pred_range = model(combine_features)
        plt.scatter(
            (pred_range * dataset_max_x_range).tolist(),
            (pred_height * dataset_max_height).tolist(),
            s=5, label=rf"{degree}$^\circ$")

    plt.legend()
    plt.xlabel("R")
    plt.ylabel("H")
    plt.title(r"$R=4H\cot \theta$")
    plt.grid(False)
    plt.savefig(args.outfile)
    plt.show()


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--out_features", type=int, default=1)
    parser.add_argument("--weight", type=str, default="model.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_t", type=float, default=90)
    parser.add_argument("--max_v", type=float, default=100)
    parser.add_argument("--outfile", type=str, default="plot/line_test.pdf")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    line_test(args)
