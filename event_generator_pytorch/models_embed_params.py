import torch
from torch import nn

# EMBEDDING PARAMS

from common.consts import *
from event_generator_pytorch.models import ToOneHot


class _EmbedParams(nn.Module):

    def __init__(self,
                 in_features: int,
                 hid_size_1: int,
                 hid_size_2: int,
                 emb_features: int,
                 device):
        super(_EmbedParams, self).__init__()

        self.emb_features = emb_features

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_size_1, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_size_1, out_features=hid_size_2, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_size_2, out_features=emb_features, bias=True),
        ).to(device=device)

        # prawdopodobnie wystarczy użyć jednej sieci, żeby kodować referencyjne warości moth1 i moth2, bo kody są takie same.

    def forward(self, x):
        # W tym miejscu trzeba rozwinąć wektory tak, by jednocześnie wszystkie linijki "odpowiadające za parametry cząstek" przeszły przez tę samą macierz embeddingu.
        emb = self.net(x.float())

        return emb


class _DembedParams(nn.Module):

    def __init__(self,
                 in_features,
                 hid_size_1,
                 hid_size_2,
                 out_features,
                 device):
        super(_DembedParams, self).__init__()

        self.out_features = out_features

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_size_1, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_size_1, out_features=hid_size_2, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hid_size_2, out_features=out_features, bias=True),
            # nn.Sigmoid(),

        ).to(device=device)

    def forward(self, x):
        deemb = self.net(x)

        return deemb


class ParamEmbedNet(nn.Module):

    def __init__(self,
                 onehot_size: int,
                 embedding_size: int,
                 device
                 ):
        super(ParamEmbedNet, self).__init__()

        self.emb = _EmbedParams(
            in_features=onehot_size,
            hid_size_1=int(onehot_size / 2),
            hid_size_2=int(onehot_size / 3),
            emb_features=embedding_size,
            device=device,
        )

        self.deemb = _DembedParams(
            in_features=embedding_size,
            hid_size_1=int(onehot_size / 3),
            hid_size_2=int(onehot_size / 2),
            out_features=onehot_size,
            device=device,
        )

    def forward(self, x):
        emb = self.emb(x)
        deemb = self.deemb(emb)

        return deemb


class ParamEmbedContainer:

    def __init__(self, device):

        self.autoenc_pdg = ParamEmbedNet(
            onehot_size=PDG_EMB_CNT,
            embedding_size=PDG_EMB_DIM,
            device=device
        )

        try:
            self.autoenc_pdg.load_state_dict(torch.load(os.path.join(parent_path(), 'data/autoenc_pdg')))
        except Exception as exp:
            raise Exception('Nie znaleziono autoenc_pdg. ' + str(exp))

        self.autoenc_part_refer = ParamEmbedNet(
            onehot_size=PART_REFER_EMB_CNT,
            embedding_size=PART_REFER_EMB_DIM,
            device=device
        )

        try:
            self.autoenc_part_refer.load_state_dict(torch.load(os.path.join(parent_path(), 'data/autoenc_part_reference')))
        except:
            raise Exception('Nie znaleziono autoenc_part_reference')

        self.toOneHot = ToOneHot(
            pdg_emb_cnt=PDG_EMB_CNT,
            stat_code_emb_cnt=STAT_CODE_EMB_CNT,
            part_refer_emb_cnt=PART_REFER_EMB_CNT,
        )

    def learn(self, value):
        for param in self.autoenc_pdg.parameters():
            param.requires_grad = value
            param.requires_grad_(value)

        for param in self.autoenc_part_refer.parameters():
            param.requires_grad = value
            param.requires_grad_(False)

    def cpu(self):
        self.autoenc_pdg = self.autoenc_pdg.cpu()
        self.autoenc_part_refer = self.autoenc_part_refer.cpu()

    def to_device(self, device: torch.device):
        self.autoenc_pdg = self.autoenc_pdg.to(device)
        self.autoenc_part_refer = self.autoenc_part_refer.to(device)

    def embed(self, x: torch.Tensor):
        in_pdg = x[:, :, 0]
        one_hot_pdg = self.toOneHot.pdg_onehot(in_pdg)
        part_pdg = self.autoenc_pdg.emb(one_hot_pdg)

        part_stat_code = x[:, :, 1].unsqueeze(dim=2).float()

        refers_1 = x[:, :, 2]
        refers_2 = x[:, :, 3]
        refers_3 = x[:, :, 4]
        refers_4 = x[:, :, 5]

        refers = torch.cat([refers_1, refers_2, refers_3, refers_4], dim=1)

        one_hot_refers = self.toOneHot.particle_reference_onehot(refers)
        part_refer: torch.Tensor = self.autoenc_part_refer.emb(one_hot_refers)

        size = int(part_refer.shape[1] / 4)
        part_refer_1 = part_refer[:, :size, :]
        part_refer_2 = part_refer[:, size:2 * size, :]
        part_refer_3 = part_refer[:, 2 * size:3 * size, :]
        part_refer_4 = part_refer[:, 3 * size:, :]

        return torch.cat([part_pdg, part_stat_code, part_refer_1, part_refer_2, part_refer_3, part_refer_4], dim=2)

    def deembed(self, x: torch.Tensor):

        start = 0
        end = self.autoenc_pdg.emb.emb_features
        one_hot_emb = self.autoenc_pdg.deemb.cpu()(x[:, :, start:end])

        start = end
        end += self.autoenc_part_refer.emb.emb_features
        one_hot_refer_1 = self.autoenc_part_refer.deemb.cpu()(x[:, :, start:end])

        start = end
        end += self.autoenc_part_refer.emb.emb_features
        one_hot_refer_2 = self.autoenc_part_refer.deemb.cpu()(x[:, :, start:end])

        start = end
        end += self.autoenc_part_refer.emb.emb_features
        one_hot_refer_3 = self.autoenc_part_refer.deemb.cpu()(x[:, :, start:end])

        start = end
        end += self.autoenc_part_refer.emb.emb_features
        one_hot_refer_4 = self.autoenc_part_refer.deemb.cpu()(x[:, :, start:end])

        one_hots = torch.cat([one_hot_emb, one_hot_refer_1, one_hot_refer_2, one_hot_refer_3, one_hot_refer_4], dim=2)

        vals = self.toOneHot.reverse(one_hots)

        return vals