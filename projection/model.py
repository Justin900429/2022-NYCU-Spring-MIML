import torch.nn as nn


class ProjectileModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ProjectileModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Sigmoid(),
            nn.Linear(hidden_features, hidden_features),
            nn.Sigmoid(),
            nn.Linear(hidden_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
