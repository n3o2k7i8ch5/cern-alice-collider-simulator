from typing import Tuple

import torch
from torch import nn

'''
class Discriminator(nn.Module):

    def __init__(self, inp: int, out: int, device):
        self.device = device
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 2048, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 512, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, out)
        ).to(device=device)

    def forward(self, x: torch.Tensor):

        x = x.flatten(start_dim=1)

        return self.net(x)

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

        self.net = nn.Sequential(
            nn.Linear(inp, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, out_features*out_length)
        ).to(device=device)


    def forward(self, x):
        batch_size = len(x)

        x = self.net(x)

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