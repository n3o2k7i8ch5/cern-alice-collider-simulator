import torch
from torch import nn


class PrtclWAEEncoder(nn.Module):
    def __init__(self, emb_features, latent_size, device):
        super(PrtclWAEEncoder, self).__init__()

        self.emb_features = emb_features
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
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

            nn.Linear(128, 128),
            nn.Dropout(.1),
            nn.Tanh(),

            nn.Linear(128, latent_size),

        ).to(device=device)

    def freeze(self, value: bool = True):
        for param in self.encoder.parameters():
            param.requires_grad = not value

    def forward(self, x: torch.Tensor):
        return self.encoder(x)
