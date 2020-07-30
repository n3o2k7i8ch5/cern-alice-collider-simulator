import torch
from torch import nn

from common.consts import *
import torch.nn.functional as func

CONV_SIZE = 100

# AUTOENCODER

def conv_out_size(in_size, padd, dial, kernel, stride):
    return int((in_size + 2 * padd - dial * (kernel - 1) - 1) / stride + 1)


def trans_conv_out_size(in_size, padd, dial, kernel, stride, out_padding):
    return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1


class AutoencoderIn(nn.Module):

    def __init__(self, samples: int, emb_features: int, latent_size: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        hid_size_1 = (PDG_EMB_CNT / 2).__round__()
        hid_size_2 = (PDG_EMB_CNT / 3).__round__()

        self.pdg_emb = nn.Sequential(
            # nn.Embedding(num_embeddings=PDG_EMB_CNT, embedding_dim=PDG_EMB_DIM)

            nn.Linear(in_features=PDG_EMB_CNT, out_features=hid_size_1, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_size_1, out_features=hid_size_2, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_size_2, out_features=PDG_EMB_DIM, bias=True),
        ).to(device=device)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=CONV_SIZE, kernel_size=(1, emb_features), padding=0),
            nn.Tanh()
        ).to(device=device)

        '''
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=CONV_SIZE, kernel_size=(1, emb_features), padding=0),
            nn.BatchNorm2d(CONV_SIZE),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=CONV_SIZE, out_channels=CONV_SIZE, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(CONV_SIZE),
            nn.LeakyReLU(0.2),
        ).to(device=device)
        '''

        in_linear_size = samples * CONV_SIZE

        hidden_2_size = int(in_linear_size - 3 * (in_linear_size - latent_size) / 4)

        self.net = nn.Sequential(
            nn.Linear(in_features=in_linear_size, out_features=hidden_2_size, bias=True),
            nn.Sigmoid()
        ).to(device=device)

        self.mu = nn.Sequential(
            nn.Linear(samples * CONV_SIZE, latent_size),
            # nn.Tanh()
        ).to(device=device)

        self.var = nn.Sequential(
            nn.Linear(samples * CONV_SIZE, latent_size),
            # nn.Sigmoid()
        ).to(device=device)

    def embed_all(self, x: torch.Tensor):
        pdg = x[:, :, 0]
        pdg_onehot = func.one_hot(pdg, num_classes=PDG_EMB_CNT).float()
        part_pdg = self.pdg_emb(pdg_onehot)

        part_stat_code = x[:, :, 1].unsqueeze(dim=2).float()

        '''
        refers_1 = x[:, :, 2]
        refers_2 = x[:, :, 3]
        refers_3 = x[:, :, 4]
        refers_4 = x[:, :, 5]

        refers = torch.cat([refers_1, refers_2, refers_3, refers_4], dim=1)

        part_refer: torch.Tensor = self.part_ref_emb(refers)

        #size = int(part_refer.shape[1] / 4)
        #part_refer_1 = part_refer[:, :size, :]
        #part_refer_2 = part_refer[:, size:2 * size, :]
        #part_refer_3 = part_refer[:, 2 * size:3 * size, :]
        #part_refer_4 = part_refer[:, 3 * size:, :]
        '''
        return torch.cat([part_pdg, part_stat_code], dim=2)

    def forward(self, x: torch.Tensor):
        cat_data = x[:, :, :2].long()
        cont_data = x[:, :, 2:]

        emb_cat = self.embed_all(cat_data)

        in_emb = torch.cat([emb_cat, cont_data], dim=2)

        in_data = in_emb.unsqueeze(dim=1)
        x = self.conv(in_data)
        x = x.flatten(start_dim=1)
        # x = self.net(x)

        lat_mu = self.mu(x)
        lat_ver = self.var(x)

        return in_emb, lat_mu, lat_ver


class AutoencoderOut(nn.Module):

    def __init__(self, latent_size: int, samples: int, emb_features: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.samples = samples
        self.emb_features = emb_features
        self.pre_conv_features = emb_features - 6

        out_linear_size = samples * self.pre_conv_features #58  # CONV_SIZE

        self.net = nn.Sequential(
            nn.Linear(latent_size, out_linear_size, bias=True),
        ).to(device=device)

        '''
        def trans_conv_out_size(in_size, padd, dial: int = 1, kernel, stride, out_padding):
            return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1
        
        '''

        self.conv = nn.Sequential(
            # in_size = A
            nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=(1, 3)),  # , padding=(0, 1)),
            # out_size = A-1 + 2 + 1 = A+2
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=1024, kernel_size=(1, 3)),  # padding=(0, 2)),
            # out_size = A+1 + 2 + 1 = A+4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=(1, 3)),  # padding=(0, 3)),
            # out_size = A+3 + 2 + 1 = A+6
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            # out_size = A+5 + 0 + 1 = A+6
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            # out_size = A+5 + 0 + 1 = A+6
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

        std = torch.exp(lat_var / 2)
        eps = torch.randn_like(std)
        lat_vec = eps.mul(std).add_(lat_mu)

        out = self.auto_out(lat_vec)
        return in_emb, out, lat_mu, lat_var

    @staticmethod
    def create(samples: int, emb_features: int, latent: int, device):
        auto_in = AutoencoderIn(samples=samples, emb_features=emb_features, latent_size=latent, device=device)
        auto_out = AutoencoderOut(latent_size=latent, samples=samples, emb_features=emb_features, device=device)

        return [auto_in, auto_out]


class PDGDeembeder(nn.Module):

    def __init__(self, pdg_embed_dim: int, pdg_count: int, device):
        super(PDGDeembeder, self).__init__()

        self.pdg_embed_dim = pdg_embed_dim
        self.pdg_count = pdg_count

        self.net = nn.Sequential(
            nn.Linear(pdg_embed_dim, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2*2048),
            nn.ReLU(),
            nn.Linear(2*2048, pdg_count),
        ).to(device)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out

    # x is of size EMB_FEATURES
    def deemb(self, x: torch.Tensor):

        pdg_embed = x[:, :, :PDG_EMB_DIM]

        other = x[:, :, PDG_EMB_DIM:]

        pdg_one_hot = self.forward(pdg_embed)

        vals = torch.argmax(pdg_one_hot, dim=2).unsqueeze(dim=2).float()

        return torch.cat([vals, other], dim=2)



