import torch
from torch import nn

CONV_SIZE = 100


# AUTOENCODER

def conv_out_size(in_size, padd, dial, kernel, stride):
    return int((in_size + 2 * padd - dial * (kernel - 1) - 1) / stride + 1)


def trans_conv_out_size(in_size, padd, dial, kernel, stride, out_padding):
    return (in_size - 1) + stride - 2 * padd + dial * (kernel - 1) + out_padding + 1


class AutoencoderIn(nn.Module):

    def __init__(self, samples: int, features: int, latent_size: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        '''
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        ).to(device=device)
        '''

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=CONV_SIZE, kernel_size=(1, features), padding=0),
            nn.BatchNorm2d(CONV_SIZE),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=CONV_SIZE, out_channels=CONV_SIZE, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(CONV_SIZE),
            nn.LeakyReLU(0.2),
        ).to(device=device)

        '''
        in_h_size = samples
        for i in range(4):
            in_h_size = conv_out_size(
                in_size=in_h_size,
                padd=2,
                dial=1,
                kernel=5,
                stride=2
            )

        in_w_size = features
        for i in range(4):
            in_w_size = conv_out_size(
                in_size=in_w_size,
                padd=2,
                dial=1,
                kernel=5,
                stride=2
            )
'''
        in_linear_size = samples * CONV_SIZE

        hidden_1_size = int(in_linear_size - (in_linear_size - latent_size) / 3)
        hidden_2_size = int(in_linear_size - 3 * (in_linear_size - latent_size) / 4)
        # hidden_2_size = int(in_linear_size - 6 * (in_linear_size - out) / 8)

        self.net = nn.Sequential(
            nn.Linear(in_features=in_linear_size, out_features=hidden_2_size, bias=True),
            nn.Sigmoid()
            # nn.LeakyReLU(0.2),
            # nn.Linear(in_features=hidden_1_size, out_features=hidden_2_size, bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden_1_size, out_features=hidden_2_size, bias=True),
            # nn.Sigmoid(),
        ).to(device=device)

        self.mu = nn.Linear(hidden_2_size, latent_size).to(device=device)
        self.var = nn.Linear(hidden_2_size, latent_size).to(device=device)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.net(x)

        lat_mu = self.mu(x)
        lat_ver = self.var(x)

        return lat_mu, lat_ver


class AutoencoderOut(nn.Module):

    def __init__(self, latent_size: int, samples: int, features: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.samples = samples
        self.features = features

        '''
        self.out_h_size = samples
        for i in range(4):
            self.out_h_size = conv_out_size(
                in_size=self.out_h_size,
                padd=2,
                dial=1,
                kernel=5,
                stride=2
            )

        self.out_w_size = features
        for i in range(4):
            self.out_w_size = conv_out_size(
                in_size=self.out_w_size,
                padd=2,
                dial=1,
                kernel=5,
                stride=2,
            )
        '''

        out_linear_size = samples * CONV_SIZE

        hidden_1_size = int(out_linear_size - 3 * (out_linear_size - latent_size) / 4)

        self.net = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=hidden_1_size, bias=True),
            # nn.BatchNorm1d(latent_size*hidden_1_size),
            # nn.LeakyReLU(0.2),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_1_size, out_features=out_linear_size, bias=True),
            # nn.LeakyReLU(0.2),
        ).to(device=device)

        # (400, 325) -> output_paddings = (1, 0), 1, (1, 0), (1, 0)
        # (300, 325) -> output_paddings = (1, 0), (0, 1), (1, 0), (1, 0)

        '''
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=2, output_padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=8, out_channels=64, kernel_size=5, padding=2, stride=2, output_padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, padding=2, stride=2, output_padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, padding=2, stride=2, output_padding=(1, 0)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),

        ).to(device=device)
        '''

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=CONV_SIZE, out_channels=CONV_SIZE, kernel_size=(1, 1), padding=0
            ),
            nn.BatchNorm2d(CONV_SIZE),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=CONV_SIZE, out_channels=1, kernel_size=(1, features), padding=0
            ),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        #        x = torch.zeros(size=x.size()).cuda()

        x = self.net(x)
        x = x.reshape(shape=(batch_size, CONV_SIZE, self.samples, 1))
        x = self.conv(x)

        # x = self.conv_out.forward(input=x, output_size=[-1, 1, self.samples, self.features],),

        return x.squeeze()


class Autoencoder(nn.Module):
    def __init__(self, auto_in: AutoencoderIn, auto_out: AutoencoderOut):
        super(Autoencoder, self).__init__()

        self.auto_in = auto_in
        self.auto_out = auto_out

    def forward(self, x: torch.Tensor):
        lat_mu, lat_var = self.auto_in(x)

        std = torch.exp(lat_var / 2)
        eps = torch.randn_like(std)
        lat_vec = eps.mul(std).add_(lat_mu)

        out = self.auto_out(lat_vec)
        return out, lat_mu, lat_var

    @staticmethod
    def create(samples: int, features: int, latent: int, device):
        auto_in = AutoencoderIn(samples=samples, features=features, latent_size=latent, device=device)
        auto_out = AutoencoderOut(latent_size=latent, samples=samples, features=features, device=device)

        return [auto_in, auto_out]