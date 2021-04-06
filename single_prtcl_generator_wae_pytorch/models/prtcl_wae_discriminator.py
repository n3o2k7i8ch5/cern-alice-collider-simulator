import torch
from torch import nn


class PrtclWAEDiscriminator(nn.Module):
    def __init__(self, latent_size, device):
        super(PrtclWAEDiscriminator, self).__init__()

        self.latent_size = latent_size

        self.discriminator = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(device=device)

    def freeze(self, value: bool = True):
        for param in self.discriminator.parameters():
            param.requires_grad = not value

    def forward(self, x: torch.Tensor):
        return self.discriminator(x).squeeze()
