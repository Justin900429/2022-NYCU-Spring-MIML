import torch.nn as nn


class ProjectileModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ProjectileModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
        )

        self.height_layer = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )

        self.range_layer = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        base = self.layers(x)
        return self.height_layer(base), self.range_layer(base)
