import torch
from torch import nn

from common.sample_norm import sample_norm


def conv_out_size(in_size, padd, dial, kernel, stride):
    return int((in_size + 2 * padd - dial * (kernel - 1) - 1) / stride + 1)


def trans_conv_out_size(in_size, padd, dial, kernel, stride, out_padding):
    return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1


'''
Takes in a batch of particles, each of size:
1 x FEATURES

Returns:
a) a batch of embedded input particles, each of size:
   1 x EMB_FEATURES

b) a batch of embedded output particles, each of size;
   1 x EMB_FEATURES

c) a batch of var and mu (for VAE KLD calculation), each of size:
   1 x PRTCL_LATENT_SPACE_SIZE
'''


class AutoencPrtcl(nn.Module):
    def __init__(self, emb_features, latent_size, device):
        super(AutoencPrtcl, self).__init__()

        self.emb_features = emb_features
        self.latent_size = latent_size

        self.mean = nn.Sequential(
            nn.Linear(emb_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_size),
        ).to(device=device)

        self.logvar = nn.Sequential(
            nn.Linear(emb_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_size),
        ).to(device=device)

        self.deembeder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, emb_features),
        ).to(device=device)

    def encode(self, x: torch.Tensor):
        lat_mean = self.mean(x)
        lat_logvar = self.logvar(x)

        return lat_mean, lat_logvar

    def decode(self, x: torch.Tensor):
        return self.deembeder(x)

    def forward(self, x: torch.Tensor):
        # VAE
        lat_mean, lat_logvar = self.encode(x)
        lat_vec = sample_norm(mean=lat_mean, logvar=lat_logvar)

        out_prtcl_emb = self.decode(lat_vec)
        return out_prtcl_emb, lat_mean, lat_logvar, lat_vec
