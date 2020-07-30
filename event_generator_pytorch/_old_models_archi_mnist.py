'''

from typing import Tuple

import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, samples: int, features: int, out: int, device):
        self.device = device
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,  # rozmiar okna (5x5)
                stride=(2, 1),  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((5 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,  # rozmiar okna (5x5)
                stride=(2, 1),  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((5 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

        ).to(device=device)

        self.conv_out = nn.Sequential(
            nn.Linear(int(64 * samples * features / 4), 1, bias=True),
            nn.Sigmoid()
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=1)
        conv_res = self.conv(x)
        res = self.conv_out(conv_res.flatten(start_dim=1))

        return res

        # return self.net(x).to(device=self.device)


def train_discriminator(dis: Discriminator, optimizer, loss, real_data, fake_data, device):
    # N = real_data.size(0)
    N = len(real_data)

    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = dis(real_data)

    # Calculate error and backpropagate
    error_real = loss(prediction_real, torch.ones(N, 1).to(device=device))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = dis(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, torch.zeros(N, 1).to(device=device))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


# defining generator class
class Generator(nn.Module):

    def __init__(self, inp, out_features, out_length, device):
        self.out_features = out_features
        self.out_length = out_length
        self.device = device
        super(Generator, self).__init__()

        self.linear_in = nn.Sequential(
            nn.Linear(inp, int(128 * out_features * out_length / 4), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=device)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 5, stride=(2, 1), padding=(2, 2), output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 128, 5, stride=(2, 1), padding=(2, 2), output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 5, stride=1, padding=2),
            nn.Tanh()

        ).to(device=device)

    def forward(self, x):
        batch_size = len(x)

        x = self.linear_in(x)
        x = x.reshape(shape=(batch_size, 128, int(self.out_length / 4), self.out_features))
        x = self.conv(x)

        return x.squeeze()


def train_generator(dis, optimizer, loss, fake_data, device):
    # N = fake_data.size(0)
    N = len(fake_data)

    # Reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    prediction = dis(fake_data)

    # Calculate error and backpropagate
    error = loss(prediction, torch.ones(N, 1).to(device=device))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error

'''