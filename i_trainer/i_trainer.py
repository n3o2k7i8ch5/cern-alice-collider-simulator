from abc import ABC, abstractmethod
import random
from typing import List, Tuple

import torch
from torch.nn import MSELoss, Module
from torch.optim import Adam, AdamW

import numpy as np
from torch.utils.data import DataLoader

import torch.nn.functional as func

from common.show_quality import show_quality
from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder
from single_prtcl_generator_vae_pytorch.models.pdg_embedder import PDGEmbedder


class ITrainer(ABC):

    @property
    @abstractmethod
    def show_feat_rng(self) -> (int, int):
        pass

    @property
    def model_save_path(self) -> str:
        return 'model.model'

    def __init__(self):
        self.device = ITrainer.get_device()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        return device

    @abstractmethod
    def load_trans_data(self):
        pass

    def prep_data(self, data: torch.Tensor, batch_size: int, valid=0.1, shuffle=True) -> (DataLoader, DataLoader):

        valid_cnt = int(len(data) * valid)

        train_vals = data[valid_cnt:, :]
        valid_vals = data[:valid_cnt, :]

        train_data_loader = DataLoader(train_vals, batch_size=batch_size, shuffle=shuffle)
        valid_data_loader = DataLoader(valid_vals, batch_size=batch_size, shuffle=shuffle)

        return train_data_loader, valid_data_loader

    def embed_data(self, data, embedders: List[PDGEmbedder]):
        return data

    def train_deembeders(self, tuples: List[Tuple[torch.Tensor, PDGEmbedder, PDGDeembedder]], epochs: int) -> List[float]:

        acc_list = []

        for tuple in tuples:
            lab_data, embedder, deembedder = tuple

            deembed_optimizer = AdamW(deembedder.parameters(), lr=1e-4)
            deemb_loss = MSELoss()

            num_classes = len(lab_data)
            real_one_hot = func.one_hot(lab_data, num_classes=num_classes).float()

            for param in embedder.parameters():
                param.requires_grad = False

            gen_one_hot = None
            for i in range(epochs):
                deembed_optimizer.zero_grad()
                embed = embedder(lab_data)
                gen_one_hot = deembedder(embed)
                err_deemb = deemb_loss(real_one_hot, gen_one_hot)
                err_deemb.backward()
                deembed_optimizer.step()

            acc = 0
            gen_one_hot = (gen_one_hot > .5).int()

            diffs = torch.eq(real_one_hot, gen_one_hot).all(dim=1).int()
            size = len(diffs)
            acc += diffs.int().sum().float()
            acc /= size

            for param in embedder.parameters():
                param.requires_grad = True

            acc_list.append(acc)

        return acc_list

    def show_deemb_quality(self, lab_data: torch.Tensor, embedder: PDGEmbedder, deembedder: PDGDeembedder):

        emb = embedder(lab_data)
        one_hot_val = deembedder(emb)
        gen_labs = torch.argmax(one_hot_val, dim=0)  # .item()  # .unsqueeze(dim=2)

        acc = torch.eq(lab_data, gen_labs).sum(dim=0).item()

        print('Deembeder acc: ' + str(acc) + '/' + str(len(lab_data)))

    @abstractmethod
    def gen_autoenc_data(self, sample_cnt: int, model: Module) -> np.array:
        pass

    def show_img_comparison(self, real_data, gen_data, log: bool = False, title: str = None):
        import matplotlib.pyplot as plt

        real_data = real_data.detach().cpu()[:50, :]
        gen_data = gen_data.detach().cpu()[:50, :]
        if log:
            real_data = real_data.log()
            gen_data = gen_data.log()

        data = torch.cat([real_data, -2*torch.ones((5, gen_data.size()[1])), gen_data], dim=0)

        plt.title(f'{title}\nup - real data :: down - fake data')
        plt.imshow(data, cmap='hot', interpolation='nearest', vmin=-1, vmax=10)
        plt.colorbar()
        plt.show()

    def show_real_gen_data_comparison(
            self,
            model,
            real_data,
            embedders,
            emb: True,
            show_histograms: bool = True,
            sample_cnt = 10_000,
            deembedder: PDGDeembedder = None,
            load_model: bool = False,
            save: bool = False):

        if load_model:
            model.load_state_dict(torch.load(self.model_save_path))

        gen_data = torch.cat(self.gen_autoenc_data(sample_cnt, model), dim=1).detach()
        real_data = real_data[:sample_cnt, :]
        if emb:
            emb_data = self.embed_data(real_data, embedders)
            if show_histograms:
                show_quality(emb_data, gen_data, feature_range=self.show_feat_rng, save=save, title='Generation comparison')
            self.show_img_comparison(emb_data, gen_data, title='Generation comparison')
        else:
            gen_data = deembedder.deemb(gen_data)
            if show_histograms:
                show_quality(real_data, gen_data, feature_range=self.show_feat_rng, save=save, title='Generation')
            self.show_img_comparison(real_data, gen_data, title='Generation')
