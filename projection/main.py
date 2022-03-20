import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from model import ProjectileModel
from dataset import make_loader


def train(args):
    if "cuda" in args.device:
        args.cuda = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Create model
    model = ProjectileModel(args.in_features, args.hidden_features, args.out_features).to(device)

    # Create dataloader
    dataloader = make_loader(args.max_t, args.max_v, args.batch_size)

    # Create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    # Start training
    total_loss = []
    for _ in tqdm(range(args.epochs)):
        batch_loss = 0.0
        for combine_features, ground_truth in dataloader:
            # Move to target device
            combine_features = combine_features.to(device)
            ground_truth = ground_truth.to(device)

            prediction = model(combine_features)
            loss = criterion(prediction, ground_truth)

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        batch_loss /= len(dataloader)
        total_loss.append(batch_loss)

    plt.plot(total_loss)
    plt.show()


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--out_features", type=int, default=2)
    parser.add_argument("--max_t", type=float, default=90)
    parser.add_argument("--max_v", type=float, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args)
