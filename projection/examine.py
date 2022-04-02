import argparse
import torch

from model import ProjectileModel
from dataset import make_loader


def examine(args):
    if "cuda" in args.device:
        args.cuda = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Create model
    model = ProjectileModel(args.in_features, args.hidden_features, args.out_features).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    # Create dataloader
    dataloader = make_loader(args.max_t, args.max_v, args.batch_size)

    dataset_max_height = dataloader.dataset.max_height
    dataset_max_x_range = dataloader.dataset.max_x_range

    total_loss = 0
    for x, (max_height, x_range) in dataloader:
        # Move to target device
        x = x.to(device)
        max_height = max_height.to(device)
        x_range = x_range.to(device)

        pred_height, pred_range = model(x)
        print("True:", (max_height.item() * dataset_max_height), (x_range.item() * dataset_max_x_range))
        print("Prediction:", pred_height.item() * dataset_max_height, pred_range.item() * dataset_max_x_range)

    total_loss /= len(dataloader)
    print(total_loss)


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
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    examine(args)
