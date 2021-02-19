import torch
from torch import nn


class PrtclWAE(nn.Module):
    def __init__(self, emb_features, latent_size, device):
        super(PrtclWAE, self).__init__()

        self.emb_features = emb_features
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(emb_features, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_size),
        ).to(device=device)

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
            nn.Linear(2048, latent_size),
        ).to(device=device)

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

    def encoder_freeze(self, value: bool = True):
        for param in self.encoder.parameters():
            param.requires_grad = not value

    def decoder_freeze(self, value: bool = True):
        for param in self.decoder.parameters():
            param.requires_grad = not value

    def discriminator_freeze(self, value: bool = True):
        for param in self.discriminator.parameters():
            param.requires_grad = not value

    def forward(self, x: torch.Tensor):
        lat_vec = self.encoder(x)
        out_prtcl_emb = self.decoder(lat_vec)
        return out_prtcl_emb, lat_vec
