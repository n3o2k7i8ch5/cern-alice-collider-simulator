import torch
from torch import nn

# GEN, DIS, ONEHOT MODELS

class Discriminator(nn.Module):

    def __init__(self, samples: int, features: int, device):
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


class Generator(nn.Module):

    def __init__(self, inp, out_features, out_length, device):
        self.out_features = out_features
        self.out_length = out_length
        self.device = device
        super(Generator, self).__init__()

        self.linear_in = nn.Sequential(
            nn.Linear(inp, out_features * out_length),
            nn.BatchNorm1d(out_features * out_length),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=self.device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=self.device)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device=self.device)

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        ).to(device=self.device)

    def forward(self, x):
        batch_size = len(x)

        x = self.linear_in(x)
        x = x.reshape(shape=(batch_size, 1, self.out_length, self.out_features))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.squeeze()

        # x = x.reshape(shape=(batch_size, self.out_length, self.out_features))

        return x


class ToOneHot:
    def __init__(self,
                 pdg_emb_cnt: int,
                 stat_code_emb_cnt: int,
                 part_refer_emb_cnt: int):
        self._pdg_emb_cnt: int = pdg_emb_cnt
        self._stat_code_emb_cnt: int = stat_code_emb_cnt
        self._part_refer_emb_cnt: int = part_refer_emb_cnt

    def pdg_onehot(self, x) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, num_classes=self._pdg_emb_cnt)

    def stat_code_onehot(self, x) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, num_classes=self._stat_code_emb_cnt)

    # +1 jest ponieważ kody są równe -1 dla referencji do nienznaej cząstki.
    def particle_reference_onehot(self, x) -> torch.Tensor:
        return torch.nn.functional.one_hot(x + 1, num_classes=self._part_refer_emb_cnt)

    def pdg_reverse(self, x) -> torch.Tensor:
        return torch.argmax(x, dim=1)

    # +1 jest ponieważ kody są równe -1 dla referencji do nienznaej cząstki.
    def particle_reference_reverse(self, x) -> torch.Tensor:
        return torch.argmax(x, dim=1) - 1

    def forward(self, x):
        # pdg_onehot: torch.Tensor = torch.nn.functional.one_hot(x[:, :, 0], num_classes=self.pdg_emb_cnt)
        pdg: torch.Tensor = self.pdg_onehot(x[:, :, 0])

        # stat_code_onehot = torch.nn.functional.one_hot(x[:, :, 1], num_classes=self.stat_code_emb_cnt)
        stat_code: torch.Tensor = self.stat_code_onehot(x[:, :, 1])

        moth1: torch.Tensor = self.particle_reference_onehot(x[:, :, 2])

        moth2: torch.Tensor = self.particle_reference_onehot(x[:, :, 3])

        daugh1: torch.Tensor = self.particle_reference_onehot(x[:, :, 4])

        daugh2: torch.Tensor = self.particle_reference_onehot(x[:, :, 5])

        return torch.cat([pdg, stat_code, moth1, moth2, daugh1, daugh2], dim=2)

    def reverse(self, x: torch.Tensor):
        start = 0

        pdg_part = x[:, :, start:start + self._pdg_emb_cnt]  # parts[0]
        start += self._pdg_emb_cnt

        stat_code_part = x[:, :, start:start + self._stat_code_emb_cnt]
        start += self._stat_code_emb_cnt

        moth1_part = x[:, :, start:start + self._part_refer_emb_cnt]
        start += self._part_refer_emb_cnt

        moth2_part = x[:, :, start:start + self._part_refer_emb_cnt]
        start += self._part_refer_emb_cnt

        daugh1_part = x[:, :, start:start + self._part_refer_emb_cnt]
        start += self._part_refer_emb_cnt

        daugh2_part = x[:, :, start:start + self._part_refer_emb_cnt]
        # start += self.daugh2_emb_cnt

        pdg = torch.argmax(pdg_part, dim=2).unsqueeze(dim=2)
        stat_code = torch.argmax(stat_code_part, dim=2).unsqueeze(dim=2)
        moth1 = torch.argmax(moth1_part, dim=2).unsqueeze(dim=2) - 1
        moth2 = torch.argmax(moth2_part, dim=2).unsqueeze(dim=2) - 1
        daugh1 = torch.argmax(daugh1_part, dim=2).unsqueeze(dim=2) - 1
        daugh2 = torch.argmax(daugh2_part, dim=2).unsqueeze(dim=2) - 1

        return torch.cat([pdg, stat_code, moth1, moth2, daugh1, daugh2], dim=2)
