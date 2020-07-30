import torch
from torch import nn

from common.consts import *

CONV_SIZE = 32


# AUTOENCODER

def conv_out_size(in_size, padd, dial, kernel, stride):
    return int((in_size + 2 * padd - dial * (kernel - 1) - 1) / stride + 1)


def trans_conv_out_size(in_size, padd, dial, kernel, stride, out_padding):
    return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1


class AutoencoderIn(nn.Module):

    def __init__(self, emb_features: int, latent_size: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        self.pdg_emb = nn.Sequential(
            nn.Embedding(num_embeddings=PDG_EMB_CNT, embedding_dim=PDG_EMB_DIM)
        ).to(device=device)

        self.net = nn.Sequential(
            nn.Linear(emb_features, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, CONV_SIZE),
            #nn.LeakyReLU(0.2),
        ).to(device=device)

        self.mu = nn.Sequential(
            nn.Linear(CONV_SIZE, latent_size),
            #nn.Tanh()
        ).to(device=device)

        self.var = nn.Sequential(
            nn.Linear(CONV_SIZE, latent_size),
            #nn.Sigmoid()
        ).to(device=device)

    def embed(self, x: torch.Tensor):

        pdg = x[:, 0]
        prtc_pdg = self.pdg_emb(pdg)
        prtc_stat_code = x[:, 1].unsqueeze(dim=1).float()

        return torch.cat([prtc_pdg, prtc_stat_code], dim=1)

    def forward(self, x: torch.Tensor):

        cat_data = x[:, :2].long()
        cont_data = x[:, 2:]

        emb_cat = self.embed(cat_data)

        in_emb = torch.cat([emb_cat, cont_data], dim=1)

        x = self.net(in_emb)

        lat_mu = self.mu(x)
        lat_var = self.var(x)

        return in_emb, lat_mu, lat_var


class AutoencoderOut(nn.Module):

    def __init__(self, latent_size: int, emb_features: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.emb_features = emb_features

        self.net = nn.Sequential(
            nn.Linear(latent_size, 64, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, emb_features, bias=True),
            nn.LeakyReLU(0.2),
        ).to(device=device)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=(1, 3)),#, padding=(0, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(1, 3)),# padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 2)),# padding=(0, 3)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
        ).to(device=device)

    def forward(self, x: torch.Tensor):

        x = self.net(x)

        return x.squeeze()


class Autoencoder(nn.Module):
    def __init__(self, auto_in: AutoencoderIn, auto_out: AutoencoderOut):
        super(Autoencoder, self).__init__()

        self.auto_in = auto_in
        self.auto_out = auto_out

    def forward(self, x: torch.Tensor):

        in_emb, lat_mu, lat_var = self.auto_in(x)

        std = torch.exp(lat_var / 2)
        eps = torch.randn_like(std)
        lat_vec = eps.mul(std).add_(lat_mu)

        out = self.auto_out(lat_vec)
        return in_emb, out, lat_mu, lat_var

    @staticmethod
    def create(emb_features: int, latent: int, device):
        auto_in = AutoencoderIn(emb_features=emb_features, latent_size=latent, device=device)
        auto_out = AutoencoderOut(latent_size=latent, emb_features=emb_features, device=device)

        return [auto_in, auto_out]