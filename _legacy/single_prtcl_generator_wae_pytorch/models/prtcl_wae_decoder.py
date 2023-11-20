import torch
from torch import nn


class PrtclWAEDecoder(nn.Module):
    def __init__(self, emb_features, latent_size, device):
        super(PrtclWAEDecoder, self).__init__()

        self.emb_features = emb_features
        self.latent_size = latent_size

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, emb_features),
        ).to(device=device)

    def freeze(self, value: bool = True):
        for param in self.decoder.parameters():
            param.requires_grad = not value

    def forward(self, x: torch.Tensor):
        return self.decoder(x)
