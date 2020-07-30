import torch
from torch import nn, optim


# AUTOENCODER

class AutoencoderIn(nn.Module):

    def __init__(self, samples: int, features: int, out: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        hidden_1_size = int(samples * features - (samples * features - out) / 3)
        hidden_2_size = int(samples * features - 2 * (samples * features - out) / 3)

        self.net = nn.Sequential(
            nn.Linear(in_features=samples * features, out_features=hidden_1_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_1_size, out_features=hidden_2_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_2_size, out_features=out, bias=True),
            nn.Sigmoid(),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        x = self.net(x)

        return x


class AutoencoderOut(nn.Module):

    def __init__(self, out: int, samples: int, features: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.samples = samples
        self.features = features

        hidden_1_size = int(samples * features - 2 * (samples * features - out) / 3)
        hidden_2_size = int(samples * features - (samples * features - out) / 3)

        self.net = nn.Sequential(
            nn.Linear(in_features=out, out_features=hidden_1_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_1_size, out_features=hidden_2_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_2_size, out_features=samples * features, bias=True),
            # nn.Sigmoid(),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        x = self.net(x)
        x = x.reshape(shape=(batch_size, self.samples, self.features))

        return x


class Autoencoder(nn.Module):
    def __init__(self, samples: int, features: int, out: int, device):
        super(Autoencoder, self).__init__()

        self.auto_in = AutoencoderIn(samples=samples, features=features, out=out, device=device)
        self.auto_out = AutoencoderOut(out=out, samples=samples, features=features, device=device)

    def forward(self, x: torch.Tensor):
        x = self.auto_in(x)
        x = self.auto_out(x)
        return x
