import torch
from torch import nn


class PDGDeembedder(nn.Module):

    def __init__(self, pdg_embed_dim: int, pdg_count: int, device):
        super(PDGDeembedder, self).__init__()

        self.pdg_embed_dim = pdg_embed_dim
        self.pdg_count = pdg_count

        self.net = nn.Sequential(
            nn.Linear(pdg_embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, pdg_count),
            nn.Tanh()
        ).to(device)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out

    # x is of size EMB_FEATURES
    def deemb(self, x: torch.Tensor):

        pdg_embed: torch.Tensor = torch.Tensor()
        other: torch.Tensor = torch.Tensor()

        if len(x.size()) == 2:
            pdg_embed = x[:, :self.pdg_embed_dim]
            other = x[:, self.pdg_embed_dim:]
        elif len(x.size()) == 3:
            pdg_embed = x[:, :, :self.pdg_embed_dim]
            other = x[:, :, self.pdg_embed_dim:]

        pdg_one_hot = self.forward(pdg_embed)

        if len(x.size()) == 2:
            vals = torch.argmax(pdg_one_hot, dim=1).unsqueeze(dim=1).float()
            return torch.cat([vals, other], dim=1)
        elif len(x.size()) == 3:
            vals = torch.argmax(pdg_one_hot, dim=2).unsqueeze(dim=2).float()
            return torch.cat([vals, other], dim=2)
