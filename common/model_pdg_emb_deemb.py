import torch
from torch import nn

from common.consts import *
import torch.nn.functional as func


class PDGDeembeder(nn.Module):

    def __init__(self, pdg_embed_dim: int, pdg_count: int, device):
        super(PDGDeembeder, self).__init__()

        self.pdg_embed_dim = pdg_embed_dim
        self.pdg_count = pdg_count

        self.net = nn.Sequential(
            nn.Linear(pdg_embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, pdg_count),
        ).to(device)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out

    # x is of size EMB_FEATURES
    def deemb(self, x: torch.Tensor):

        pdg_embed: torch.Tensor = torch.Tensor()
        other: torch.Tensor = torch.Tensor()

        if len(x.size()) == 2:
            pdg_embed = x[:, :PDG_EMB_DIM]
            other = x[:, PDG_EMB_DIM:]
        elif len(x.size()) == 3:
            pdg_embed = x[:, :, :PDG_EMB_DIM]
            other = x[:, :, PDG_EMB_DIM:]

        pdg_one_hot = self.forward(pdg_embed)

        if len(x.size()) == 2:
            vals = torch.argmax(pdg_one_hot, dim=1).unsqueeze(dim=1).float()
            return torch.cat([vals, other], dim=1)
        elif len(x.size()) == 3:
            vals = torch.argmax(pdg_one_hot, dim=2).unsqueeze(dim=2).float()
            return torch.cat([vals, other], dim=2)


class PDGEmbeder(nn.Module):
    def __init__(self, pdg_embed_dim: int, pdg_count: int, device):
        super(PDGEmbeder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(pdg_count, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, pdg_embed_dim),
            nn.Tanh(),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        return self.net(x)


def show_deemb_quality(embeder: PDGEmbeder, deembeder: PDGDeembeder, device):
    prtcl_idxs = torch.tensor(particle_idxs(), device=device)

    pdg_onehot = func.one_hot(
        prtcl_idxs,
        num_classes=PDG_EMB_CNT
    ).float()
    emb = embeder(pdg_onehot)
    one_hot_val = deembeder(emb)
    gen_idxs = torch.argmax(one_hot_val, dim=0)  # .item()  # .unsqueeze(dim=2)

    acc = (torch.eq(prtcl_idxs, gen_idxs) == True).sum(dim=0).item()

    print('Deembeder acc: ' + str(acc) + '/' + str(len(prtcl_idxs)))