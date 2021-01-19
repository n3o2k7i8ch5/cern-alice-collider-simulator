import torch
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.nn.functional as func
import numpy as np

from common.consts import PRTCL_LATENT_SPACE_SIZE, EMB_FEATURES, PDG_EMB_DIM, particle_idxs, parent_path, CAT_FEATURES, \
    CONT_FEATURES
from common.device import get_device
from common.model_pdg_emb_deemb import PDG_EMB_CNT, BATCH_SIZE, PARTICLE_DIM, PDGDeembedder, PDGEmbedder
from common.models_prtc_embed_autoenc import AutoencPrtcl
from common.show_quality import show_lat_histograms, show_quality
from single_particle_generator_pytorch.load_data import load_data

import pandas as pd

from single_particle_generator_pytorch.train_deembeder import train_deembeder


class Trainer:
    AUTOENC_SAVE_PATH = parent_path() + 'data/single_prtc_autoenc.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'

    SHOW_FEAT_RANGE = (-7, -4)

    def __init__(self):
        self.device = get_device()

    def load_data(self) -> torch.Tensor:
        data = load_data()
        # tensor = torch.tensor(data, device=self.device)
        return data

    def prep_data(self, data: torch.Tensor, batch_size: int, valid=0.1, shuffle=True) -> (DataLoader, DataLoader):

        valid_cnt = int(len(data) * valid)

        train_vals = data[valid_cnt:, :]
        valid_vals = data[:valid_cnt, :]

        train_data_loader = DataLoader(train_vals, batch_size=batch_size, shuffle=shuffle)
        valid_data_loader = DataLoader(valid_vals, batch_size=batch_size, shuffle=shuffle)

        return train_data_loader, valid_data_loader

    def loss(self, input_x, output_x, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (
    Variable, Variable, Variable):
        mse_loss = MSELoss()(input_x, output_x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return mse_loss + kld_loss * 1e-4, mse_loss, kld_loss

    def embed_data(self, embeder: PDGEmbedder, data):
        cat_data = data[:, :2].long()
        cont_data = data[:, 2:]

        embeder = embeder.to(data.device)

        pdg = cat_data[:, 0]
        pdg_onehot = func.one_hot(pdg, num_classes=PDG_EMB_CNT).float()
        prtc_pdg = embeder(pdg_onehot)

        prtc_stat_code = cat_data[:, 1].unsqueeze(dim=1).float()

        return torch.cat([prtc_pdg, prtc_stat_code, cont_data], dim=1)

    def create_autoenc(self) -> AutoencPrtcl:
        return AutoencPrtcl(
            emb_features=EMB_FEATURES,
            latent_size=PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

    def create_embedder(self) -> PDGEmbedder:
        return PDGEmbedder(PDG_EMB_DIM, PDG_EMB_CNT, self.device)

    def create_deembedder(self) -> PDGDeembedder:
        return PDGDeembedder(PDG_EMB_DIM, PDG_EMB_CNT, self.device)

    def create_train_deembedder(self, embedder, epochs):
        deembedder = self.create_deembedder()
        train_deembeder(
            embedder=embedder, deembedder=deembedder, epochs=epochs, device=self.device)
        return deembedder

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(BATCH_SIZE) +
              ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
              ', EPOCHS: ' + str(epochs) +
              ', PRTCL_LATENT_SPACE_SIZE: ' + str(PRTCL_LATENT_SPACE_SIZE)
              )

        autoenc = self.create_autoenc()
        autoenc_optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.00002)

        embedder = self.create_embedder()
        deembedder: PDGDeembedder = PDGDeembedder(PDG_EMB_DIM, PDG_EMB_CNT, self.device)

        if load:
            print('LOADING MODEL STATES...')
            autoenc.load_state_dict(torch.load(Trainer.AUTOENC_SAVE_PATH))
            embedder.load_state_dict(torch.load(Trainer.PDG_EMBED_SAVE_PATH))
            deembedder.load_state_dict(torch.load(Trainer.PDG_DEEMBED_SAVE_PATH))

        print('AUTOENCODER')
        print(autoenc)
        print('EMBEDDER')
        print(embedder)
        print('DEEMBEDDER')
        print(deembedder)

        _data = self.load_data()
        data_train, data_valid = self.prep_data(_data, batch_size=BATCH_SIZE, valid=0.1)

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):
                autoenc_optimizer.zero_grad()

                real_data: torch.Tensor = batch.to(self.device).detach()

                emb_data = self.embed_data(embedder, real_data)

                gen_data, lat_mean, lat_logvar, lat_vec = autoenc(emb_data)

                loss, mse_loss, kld_loss = self.loss(
                    input_x=emb_data,
                    output_x=gen_data,
                    lat_mean=lat_mean,
                    lat_logvar=lat_logvar)

                loss.backward()
                autoenc_optimizer.step()

                train_deembeder(deembedder=deembedder, embedder=embedder, epochs=1, device=self.device)

                if n_batch % 500 == 0:
                    show_lat_histograms(lat_mean=lat_mean, lat_logvar=lat_logvar)
                    self.show_deemb_quality(embedder, deembedder)
                    valid_loss = self._valid_loss(autoenc, embedder, data_valid)

                    show_quality(emb_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE, save=True)
                    self.show_real_gen_data_comparison(autoenc, embedder, emb=False, deembedder=deembedder, save=True)
                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'train loss: {"{:.6f}".format(round(loss.item(), 6))} :: '
                        f'kld loss: {"{:.6f}".format(round(kld_loss.item(), 6))} :: '
                        f'mse loss: {"{:.6f}".format(round(mse_loss.item(), 6))} :: '
                        f'valid loss: {"{:.6f}".format(round(valid_loss, 6))}'
                    )

            self._save_models(autoenc, embedder, deembedder)

        return autoenc, embedder, deembedder

    def _save_models(self, autoenc, embedder, deembedder):
        print('Saving autoencoder model')
        torch.save(autoenc.state_dict(), Trainer.AUTOENC_SAVE_PATH)
        print('Saving embed model')
        torch.save(embedder.state_dict(), Trainer.PDG_EMBED_SAVE_PATH)
        print('Saving deembedder model')
        torch.save(deembedder.state_dict(), Trainer.PDG_DEEMBED_SAVE_PATH)

    def _valid_loss(self, autoenc, embedder, valid_data_loader) -> float:
        loss = 0
        criterion = MSELoss()

        for batch_data in valid_data_loader:
            emb_data = self.embed_data(embedder, batch_data.to(self.device))
            out_prtcl_emb, lat_mean, lat_vec, lat_vec = autoenc(emb_data)
            train_loss = criterion(out_prtcl_emb, emb_data)

            loss += train_loss.item()

        loss /= len(valid_data_loader)

        return loss

    def show_deemb_quality(self, embedder, deembedder):
        prtcl_idxs = torch.tensor(particle_idxs(), device=self.device)

        pdg_onehot = func.one_hot(
            prtcl_idxs,
            num_classes=PDG_EMB_CNT
        ).float()

        emb = embedder(pdg_onehot)
        one_hot_val = deembedder(emb)
        gen_idxs = torch.argmax(one_hot_val, dim=0)  # .item()  # .unsqueeze(dim=2)

        acc = torch.eq(prtcl_idxs, gen_idxs).sum(dim=0).item()

        print('Deembeder acc: ' + str(acc) + '/' + str(len(prtcl_idxs)))

    def gen_autoenc_data(self, sample_cnt, autoenc):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = autoenc.decode(rand_input).detach()
        return generated_data

    def show_img_comparison(self, data, title='Some data...'):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(title)
        plt.imshow(data.detach().cpu().log(), cmap='hot', interpolation='nearest', vmin=-10, vmax=10)
        plt.legend()
        plt.show()
        plt.ion()
        plt.pause(0.001)

    def show_real_gen_data_comparison(
            self,
            autoenc,
            embedder,
            emb: True,
            deembedder: PDGDeembedder = None,
            load_model: bool = False,
            save: bool = False
    ):

        if load_model:
            autoenc.load_state_dict(torch.load(Trainer.AUTOENC_SAVE_PATH))

        gen_data = self.gen_autoenc_data(10_000, autoenc)
        real_data = self.load_data()[:10_000]
        if emb:
            emb_data = self.embed_data(embedder, real_data)
            show_quality(emb_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE, save=save,
                         title='Generation comparison')
        else:
            gen_data = deembedder.deemb(gen_data)
            show_quality(real_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE, save=save,
                         title='Generation comparison')
            self.show_img_comparison(real_data[:30, :], 'Real data')
            self.show_img_comparison(gen_data[:30, :], 'Gen data')

