import torch
from torch import nn


class PrtclWGANDiscriminator(nn.Module):
    def __init__(self, emb_features, device):
        super(PrtclWGANDiscriminator, self).__init__()

        self.emb_features = emb_features

        self.__net = nn.Sequential(
            nn.Linear(emb_features, 1024),
            nn.Dropout(.1),
            nn.Tanh(),

            nn.Linear(1024, 512),
            nn.Dropout(.1),
            nn.Tanh(),

            nn.Linear(512, 256),
            nn.Dropout(.1),
            nn.Tanh(),

            nn.Linear(256, 128),
            nn.Dropout(.1),
            nn.Tanh(),

            nn.Linear(128, 16),
            nn.Tanh(),
            nn.Linear(16, 1),

        ).to(device=device)

    def forward(self, x: torch.Tensor):
        return self.__net(x).squeeze()
