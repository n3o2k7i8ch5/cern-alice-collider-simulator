import torch
from torch import nn
from torch.nn.functional import one_hot


class PDGEmbedder(nn.Module):
    def __init__(self, pdg_count: int, pdg_embed_dim: int, device):
        super(PDGEmbedder, self).__init__()

        self.pdg_count = pdg_count
        self.net = nn.Sequential(
            nn.Linear(pdg_count, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, pdg_embed_dim),
            nn.Tanh(),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        onehot = one_hot(x, self.pdg_count).float()
        return self.net(onehot)

    def forward_onehot(self, x_onehot: torch.Tensor):
        return self.net(x_onehot)