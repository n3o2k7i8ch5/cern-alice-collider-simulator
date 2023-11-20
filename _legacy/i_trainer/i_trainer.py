import os
from abc import ABC, abstractmethod
from math import ceil, floor
from typing import List, Tuple

import torch
from torch.nn import Module
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import DataLoader

from scipy import stats

from _legacy.common.consts import FEATURES, PDG_EMB_CNT, PDG_EMB_DIM

from _legacy import config
from _legacy.common.models.pdg_deembedder import PDGDeembedder
from _legacy.common.models.pdg_embedder import PDGEmbedder


class RealGenDataSizeException(Exception):
    def __init__(self, real_data_size: int, gen_data_size: int):
        super(RealGenDataSizeException, self).__init__(
            f'Real data size: {real_data_size}, gen data size: {gen_data_size}.')


class ITrainer(ABC):

    @property
    def pdg_emb_cnt(self) -> int:
        return PDG_EMB_CNT

    @property
    def pdg_emb_dim(self) -> int:
        return PDG_EMB_DIM

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

    def create_embedder(self):
        return PDGEmbedder(pdg_count=self.pdg_emb_cnt, pdg_embed_dim=self.pdg_emb_dim, device=self.device)

    def create_deembedder(self) -> PDGDeembedder:
        return PDGDeembedder(self.pdg_emb_dim, self.pdg_emb_cnt, self.device)

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

        data = torch.cat([real_data, -2 * torch.ones((5, gen_data.size()[1])), gen_data], dim=0)

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
            sample_cnt=10_000,
            bin_cnt: int = 100,
            hist_range: Tuple = (-2, 2),
            deembedder: PDGDeembedder = None,
            load_model: bool = False,

            save: bool = False,
            epoch: int = None,
            batch: int = None,
    ) -> [float, float]:

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

        fig, subplots = plt.subplots(nrows=ceil(len(attr_idxs) / 2), ncols=2, sharex='all')

        fig.subplots_adjust(wspace=.35, hspace=.35, top=.8, bottom=.17)
        fig.suptitle('Porównanie histogramów wybranych atrybutów\ndanych rzeczywistych i syntetycznych.', size=14)

        err_kld = 0
        err_wass = 0

        subplot_handlers = []

        for i, attr_idx in enumerate(attr_idxs):
            _real_data = real_data[:, attr_idx].flatten().cpu()
            _gen_data = gen_data[:, attr_idx].flatten().cpu()

            subplots[floor(i / 2)][i % 2].set_title(f'Atrybut: {attr_idx}')
            handler = subplots[floor(i / 2)][i % 2].hist([_real_data, _gen_data],
                                               range=hist_range,
                                               stacked=False,
                                               bins=bin_cnt,
                                               histtype='stepfilled',
                                               label=['real data', 'synthetic data'],
                                               color=['blue', 'red'],
                                               alpha=0.5,
                                               )

            subplot_handlers.append(handler)

            # KL_DIV
            min_val = min(min(_real_data), min(_gen_data))
            max_val = max(max(_real_data), max(_gen_data))

            steps = np.linspace(start=min_val, stop=max_val, num=bin_cnt)

            real_data_hist, _ = np.histogram(
                _real_data,
                bins=steps)

            gen_data_hist, _ = np.histogram(
                _gen_data,
                bins=steps)

            real_log = np.log(real_data_hist + 1)

            gen_log = np.log(gen_data_hist + 1)

            diff_log = abs(real_log - gen_log)

            err_kld += sum(real_data_hist * diff_log + gen_data_hist * diff_log)

            dists = [i for i in range(len(real_data_hist))]
            err_wass += stats.wasserstein_distance(dists, dists, real_data_hist, gen_data_hist)

        fig.legend(subplot_handlers, labels=['Dane syntetyczne', 'Dane rzeczywiste'], loc="lower right")

        if save:
            if not os.path.isdir(config.plot_folder):
                os.mkdir(config.plot_folder)

            title = 'reprod'
            fig.savefig(f'{config.plot_folder}/hists_{epoch}_{batch}_{title}.png')

        fig.show()

        return err_kld / len(attr_idxs), err_wass

    @staticmethod
    def show_comp_hists(
            real_data: np.ndarray,
            fake_data: np.ndarray,
            attr_idx: int,
            reprod: bool,
            xlabel='',
            ylabel='',
            bin_cnt: int = 100,
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
                 bins=bin_cnt,
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

    @staticmethod
    def evaluate_gen_model(real_data_batch: torch.Tensor, gen_data_batch: torch.Tensor, bin_cnt: int = 100) -> float:

        err = 0

        attr_cnt = real_data_batch.size()[1]
        for i in range(attr_cnt):

            if FEATURES - 6 != i:
                continue

            real_data = real_data_batch[:, i].flatten()
            gen_data = gen_data_batch[:, i].flatten()

            if len(real_data) != len(gen_data):
                raise RealGenDataSizeException(len(real_data), len(gen_data))

            real_data = (real_data/len(real_data)).detach().numpy()
            gen_data = (gen_data/len(gen_data)).detach().numpy()

            min_val = min(min(real_data), min(gen_data))
            max_val = max(max(real_data), max(gen_data))

            steps = np.linspace(start=min_val, stop=max_val, num=bin_cnt)

            real_data_hist, _ = np.histogram(
                real_data,
                bins=steps)

            gen_data_hist, _ = np.histogram(
                gen_data,
                bins=steps)

            real_log = np.log(real_data_hist + 1)
            print(f'real_log\n{real_data_hist}\n\n')

            gen_log = np.log(gen_data_hist + 1)
            #print(f'gen_log\n{gen_log}\n\n')

            diff_log = abs(real_log - gen_log)
            #print(f'diff_log\n{diff_log}\n\n')


            err += sum(real_data_hist*diff_log + gen_data_hist*diff_log)

        return err/attr_cnt
