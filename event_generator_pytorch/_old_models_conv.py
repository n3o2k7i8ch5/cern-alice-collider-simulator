from typing import Tuple

import torch
from torch import nn

'''
class Discriminator(nn.Module):

    def __init__(self, inp: int, out: int, device):
        self.device = device
        super(Discriminator, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,  # rozmiar okna (5x5)
                stride=1,  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((3 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(8),
        ).to(device=device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,  # rozmiar okna (5x5)
                stride=1,  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((3 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            #nn.AdaptiveMaxPool2d(output_size=(x.shape[0], x.shape[1], x.shape[2] / 2, x.shape[3])),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
        ).to(device=device)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,  # rozmiar okna (5x5)
                stride=1,  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((5 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
        ).to(device=device)

        self.linear1 = nn.Sequential(
            nn.Linear(int(inp*16/(2*2*2)), 32, bias=True),
            nn.BatchNorm1d(32, 0.8),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(0.1)
        ).to(device=device)

        self.linear_out = nn.Sequential(
            nn.Linear(32, out),
            nn.Sigmoid()
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        #x = torch.flatten(x, start_dim=1)

        x = x.unsqueeze(dim=1)

        x = self.conv1(x)
        #x = nn.AdaptiveMaxPool2d(output_size=(x.shape[0], x.shape[1], x.shape[2]/2, x.shape[3])),
        x = self.conv2(x)
        #x = nn.AdaptiveMaxPool2d(output_size=(x.shape[0], x.shape[1], x.shape[2]/2, x.shape[3])),
        x = self.conv3(x)

        x = x.flatten(start_dim=1)

        x = self.linear1(x)
        return self.linear_out(x)

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

        self.linear_in = nn.Sequential(nn.Linear(inp, out_features * out_length)).to(device=device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=self.device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=self.device)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=self.device)

        self.linear_out = nn.Sequential(
            nn.Linear(
                16 * out_features * out_length,
                out_features * out_length),
        ).to(device=self.device)

    def forward(self, x):
        batch_size = len(x)

        x = self.linear_in(x)
        x = x.reshape(shape=(batch_size, 1, self.out_features, self.out_length))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)
        x = self.linear_out(x)

        x = x.reshape(shape=(batch_size, self.out_length, self.out_features))

        return x


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