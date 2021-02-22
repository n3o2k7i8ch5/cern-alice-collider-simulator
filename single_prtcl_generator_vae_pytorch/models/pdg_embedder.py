import torch
from torch import nn
from torch.nn.functional import one_hot


class PDGEmbedder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device):
        super(PDGEmbedder, self).__init__()

        self.num_embeddings = num_embeddings
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, embedding_dim),
            nn.Tanh(),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        onehot = one_hot(x, self.num_embeddings).float()
        return self.net(onehot)
