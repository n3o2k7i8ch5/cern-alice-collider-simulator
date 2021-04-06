import os
from abc import ABC, abstractmethod
from math import ceil, floor
from typing import List, Tuple

import torch
from torch.nn import MSELoss, Module
from torch.optim import AdamW
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import DataLoader

import torch.nn.functional as func

from common.show_quality import show_quality
from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder
from single_prtcl_generator_vae_pytorch.models.pdg_embedder import PDGEmbedder

import config


class ITrainer(ABC):

    '''
    @property
    @abstractmethod
    def show_feat_rng(self) -> (int, int):
        pass
    '''

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

    def print_deemb_quality(self, lab_data: torch.Tensor, embedder: PDGEmbedder, deembedder: PDGDeembedder):

        emb = embedder(lab_data)
        one_hot_val = deembedder(emb)
        gen_labs = torch.argmax(one_hot_val, dim=0)

        acc = torch.eq(lab_data, gen_labs).sum(dim=0).item()

        print('Deembeder acc: ' + str(acc) + '/' + str(len(lab_data)))

    @abstractmethod
    def gen_autoenc_data(self, sample_cnt: int, model: Module) -> torch.Tensor or List[torch.Tensor]:
        pass

    @staticmethod
    def show_heatmaps(
            real_data,
            gen_data,
            reprod: bool, 
            log: bool = False,

            save: bool = False,
            epoch: int = None,
            batch: int = None,
    ):

        real_data = real_data.detach().cpu()[:50, :]
        gen_data = gen_data.detach().cpu()[:50, :]
        if log:
            real_data = real_data.log()
            gen_data = gen_data.log()

        data = torch.cat([real_data, -2*torch.ones((5, gen_data.size()[1])), gen_data], dim=0)

        plt.title(f'{"reprod" if reprod else "gener"}\nup - real data :: down - fake data')
        plt.imshow(data, cmap='hot', interpolation='nearest', vmin=-1, vmax=5)
        plt.colorbar()

        if save:
            if not os.path.isdir(config.plot_folder):
                os.mkdir(config.plot_folder)

            title = 'reprod' if reprod else 'gener'
            plt.savefig(f'{config.plot_folder}/_heatmap_{epoch}_{batch}_{title}.png')

        plt.show()

    def gen_show_comp_hists(
            self,
            model,
            all_real_data,
            attr_idxs: list,
            embedders: List,
            emb: True,
            sample_cnt = 10_000,
            range: Tuple = (-2, 2),
            deembedder: PDGDeembedder = None,
            load_model: bool = False,

            save: bool = False,
            epoch: int = None,
            batch: int = None,
    ):

        if load_model:
            model.load_state_dict(torch.load(self.model_save_path))

        gen_data = self.gen_autoenc_data(sample_cnt, model)
        if gen_data is Tuple:
            gen_data = torch.cat(gen_data, dim=1).detach()

        real_data = all_real_data[:sample_cnt, :]
        if emb:
            real_data = self.embed_data(real_data, embedders)
        else:
            gen_data = deembedder.deemb(gen_data)

        fig, subplots = plt.subplots(nrows=ceil(len(attr_idxs)/2), ncols=2)

        for i, attr_idx in enumerate(attr_idxs):
            _real_data = real_data[:, attr_idx].flatten().cpu()
            _gen_data = gen_data[:, attr_idx].flatten().cpu()

            subplots[floor(i/2)][i%2].set_title(f'attr: {attr_idx}')
            subplots[floor(i/2)][i%2].hist([_real_data, _gen_data],
                     range=range,
                     stacked=False,
                     bins=100,
                     histtype='stepfilled',
                     label=['real data', 'synthetic data'],
                     color=['blue', 'red'],
                     alpha=0.5
                    )

        if save:
            if not os.path.isdir(config.plot_folder):
                os.mkdir(config.plot_folder)

            title = 'reprod'
            fig.savefig(f'{config.plot_folder}/hists_{epoch}_{batch}_{title}.png')

        fig.show()

    @staticmethod
    def show_comp_hists(
            real_data: np.ndarray,
            fake_data: np.ndarray,
            attr_idx: int,
            reprod: bool,
            xlabel='',
            ylabel='',
            range: Tuple = None,
            save: bool = None,
            epoch: int = None,
            batch: int = None,

    ):
        plt.style.use('ggplot')
        plt.title(f'Reproduction hists attr {attr_idx}' if reprod else f'Generated data hists attr {attr_idx}')

        real_data = real_data.flatten()
        fake_data = fake_data.flatten()

        plt.hist([real_data, fake_data],
                 range=range,
                 stacked=False,
                 bins=100,
                 histtype='stepfilled',
                 label=['real data', 'synthetic data'],
                 color=['blue', 'red'],
                 alpha=0.5
                 )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()

        if save:
            if not os.path.isdir(config.plot_folder):
                os.mkdir(config.plot_folder)

            title = 'reprod' if reprod else 'gener'
            plt.savefig(f'{config.plot_folder}/hists_{attr_idx}_{epoch}_{batch}_{title}.png')

        plt.show()