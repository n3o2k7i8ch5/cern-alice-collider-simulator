'''
import torch
from torch import nn


class Discriminator(nn.Module):


    def __init__(self, samples: int, features: int, out: int, device):
        self.device = device
        super(Discriminator, self).__init__()

        f_kernel_size = 40
        f_stride = 5

        ###
        ###
        self.avg_f = nn.Sequential(
            nn.AvgPool2d(kernel_size=(f_kernel_size, 1), stride=(f_stride, 1)),
        ).to(device=device)

        self.avg_f_out = nn.Sequential(
            nn.Linear(
                #features*(int(samples / (kernel_size/stride)) + 1),
                (1 + int((samples - f_kernel_size)/f_stride)) * features,
                512,
                bias=True),
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()
        ).to(device=device)
        ###
        ###

        ###
        ###
        self.avg_s = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, features)),
        ).to(device=device)

        self.avg_s_out = nn.Sequential(
            nn.Linear(
                # features*(int(samples / (kernel_size/stride)) + 1),
                samples,
                512,
                bias=True),
            # nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()
        ).to(device=device)
        ###
        ###

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,  # rozmiar okna (5x5)
                stride=1,  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((3 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.BatchNorm2d(8),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,  # rozmiar okna (5x5)
                stride=1,  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((3 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.BatchNorm2d(16),
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=1,  # rozmiar okna (5x5)
                stride=1,  # odległość każdorazowego przesunięcia kernela po obrazie.
                padding=int((1 - 1) / 2)  # liczba zer dookoła obrazka
            ),
            nn.BatchNorm2d(1),
        ).to(device=device)

        self.conv_out = nn.Sequential(
            nn.Linear(samples*features, 2048, bias=True),
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512, bias=True),
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128, bias=True),
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, out),
            nn.Sigmoid()
        ).to(device=device)

    def forward(self, x: torch.Tensor):

        avg_f = self.avg_f(x)
        avg_f = avg_f.flatten(start_dim=1)
        avg_f_res = self.avg_f_out(avg_f)

        avg_s = self.avg_s(x)
        avg_s = avg_s.flatten(start_dim=1)
        avg_s_res = self.avg_s_out(avg_s)

        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        res = self.conv_out(x)

        return 0.8 * res + 0.1*avg_f_res + 0.1*avg_s_res

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
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512, bias=True),
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024, bias=True),
            #nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
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
