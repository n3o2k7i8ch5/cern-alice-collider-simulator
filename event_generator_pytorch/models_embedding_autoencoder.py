import torch
from torch import nn

from common.consts import *
import torch.nn.functional as func

from common.gen_v_latent import get_latentt
from common.model_pdg_emb_deemb import PDGEmbedder

CONV_SIZE = 10

# AUTOENCODER

def conv_out_size(in_size, padd, dial, kernel, stride):
    return int((in_size + 2 * padd - dial * (kernel - 1) - 1) / stride + 1)


def trans_conv_out_size(in_size, padd, dial, kernel, stride, out_padding):
    return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1


class AutoencoderIn(nn.Module):

    def __init__(self, samples: int, emb_features: int, latent_size: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        self.pdg_emb = PDGEmbedder(PDG_EMB_DIM, PDG_EMB_CNT, device)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=CONV_SIZE, kernel_size=(1, emb_features), padding=0),
            nn.Tanh()
        ).to(device=device)

        self.mu = nn.Sequential(
            nn.Linear(samples * CONV_SIZE, latent_size),
            # nn.Tanh()
        ).to(device=device)

        self.var = nn.Sequential(
            nn.Linear(samples * CONV_SIZE, latent_size),
            # nn.Tanh()
        ).to(device=device)

    def embed_all(self, x: torch.Tensor):
        pdg = x[:, :, 0]
        pdg_onehot = func.one_hot(pdg, num_classes=PDG_EMB_CNT).float()
        part_pdg = self.pdg_emb(pdg_onehot)

        part_stat_code = x[:, :, 1].unsqueeze(dim=2).float()

        return torch.cat([part_pdg, part_stat_code], dim=2)

    def forward(self, x: torch.Tensor):
        cat_data = x[:, :, :2].long()
        cont_data = x[:, :, 2:]

        emb_cat = self.embed_all(cat_data)

        in_emb = torch.cat([emb_cat, cont_data], dim=2)

        in_data = in_emb.unsqueeze(dim=1)
        x = self.conv(in_data)
        x = x.flatten(start_dim=1)

        lat_mu = self.mu(x)
        lat_ver = self.var(x)

        return in_emb, lat_mu, lat_ver

class AutoencoderOut(nn.Module):

    def __init__(self, latent_size: int, samples: int, emb_features: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.samples = samples
        self.emb_features = emb_features
        self.pre_conv_features = emb_features - 16

        out_linear_size = samples * self.pre_conv_features #58  # CONV_SIZE

        self.net = nn.Sequential(
            nn.Linear(latent_size, latent_size, bias=True),
            nn.Tanh(),
            nn.Linear(latent_size, 3*latent_size, bias=True),
            nn.Tanh(),
            nn.Linear(3*latent_size, out_linear_size, bias=True),
            nn.Tanh(),
        ).to(device=device)

        '''
        def trans_conv_out_size(in_size, padd, dial: int = 1, kernel, stride, out_padding):
            return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1
        '''

        self.conv = nn.Sequential(
            # in_size = A
            nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(1, 3)),  # , padding=(0, 1)),
            # out_size = A-1 + 2 + 1 = A+2
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(1, 3)),  # padding=(0, 2)),
            # out_size = A+1 + 2 + 1 = A+4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 3)),  # padding=(0, 3)),
            # out_size = A+3 + 2 + 1 = A+6
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(1, 3)),
            # out_size = A+3 + 2 + 1 = A+8
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(1, 3)),
            # out_size = A+3 + 2 + 1 = A+10
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 3)),
            # out_size = A+5 + 2 + 1 = A+12
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3)),
            # out_size = A+5 + 2 + 1 = A+14
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3)),
            # out_size = A+5 + 2 + 1 = A+16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 1)),
            # out_size = A+5 + 0 + 1 = A+14
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        batch_size = len(x)

        x = self.net(x)
        x = x.reshape(shape=(batch_size, 1, self.samples, self.pre_conv_features))
        x = self.conv(x)

        # x = self.conv_out.forward(input=x, output_size=[-1, 1, self.samples, self.features],),

        return x.squeeze()


class Autoencoder(nn.Module):
    def __init__(self, auto_in: AutoencoderIn, auto_out: AutoencoderOut):
        super(Autoencoder, self).__init__()

        self.auto_in = auto_in
        self.auto_out = auto_out

    def forward(self, x: torch.Tensor):
        in_emb, lat_mu, lat_var = self.auto_in(x)

        lat_vec = get_latentt(lat_mu, lat_var)

        out = self.auto_out(lat_vec)
        return in_emb, out, lat_mu, lat_var

    @staticmethod
    def create(samples: int, emb_features: int, latent: int, device):
        auto_in = AutoencoderIn(samples=samples, emb_features=emb_features, latent_size=latent, device=device)
        auto_out = AutoencoderOut(latent_size=latent, samples=samples, emb_features=emb_features, device=device)

        return [auto_in, auto_out]