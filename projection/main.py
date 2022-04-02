import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from model import ProjectileModel
from dataset import make_loader


def train(args):
    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Create model
    model = ProjectileModel(args.in_features, args.hidden_features, args.out_features).to(device)

    # Create dataloader
    dataloader = make_loader(args.max_t, args.max_v, args.batch_size)

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
    criterion = torch.nn.MSELoss()

    # Start training
    total_loss = []

    progress_bar = tqdm(range(args.epochs))
    for _ in progress_bar:
        batch_loss = 0.0
        for x, (max_height, x_range) in dataloader:
            # Move to target device
            x = x.to(device)
            max_height = max_height.to(device)
            x_range = x_range.to(device)

            pred_height, pred_range = model(x)
            height_loss = criterion(pred_height, max_height)
            range_loss = criterion(pred_range, x_range)
            loss = height_loss + range_loss

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        scheduler.step()
        batch_loss /= len(dataloader)
        progress_bar.set_postfix({"loss": batch_loss})
        total_loss.append(batch_loss)

    torch.save(model.state_dict(), "model.pt")

    plt.plot(total_loss)
    plt.show()


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--out_features", type=int, default=1)
    parser.add_argument("--max_t", type=float, default=90)
    parser.add_argument("--max_v", type=float, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args)
