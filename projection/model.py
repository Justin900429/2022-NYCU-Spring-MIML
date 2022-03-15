import torch.nn as nn


class ProjectileModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ProjectileModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.001)

    def forward(self, x):
        return self.layers(x)
