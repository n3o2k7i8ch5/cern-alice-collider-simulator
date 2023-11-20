import pickle
from typing import List

import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import numpy as np

from _legacy.common.consts import EMB_FEATURES, parent_path, PARTICLE_DIM, particle_idxs, FEATURES
from _legacy.single_prtcl_generator_vae_pytorch.models.prtcl_vae import PrtclVAE
from _legacy.i_trainer.i_trainer import ITrainer
from _legacy.i_trainer.load_data import load_data


class Trainer(ITrainer):
    BATCH_SIZE = 128*8

    PRTCL_LATENT_SPACE_SIZE = 12
    LR = 5e-5

    errs_kld = []
    errs_wass = []

    AUTOENC_SAVE_PATH = parent_path() + 'data/single_prtc_autoenc.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'
    ERRS_SAVE_PATH = parent_path() + 'data/errs_vae.model'

    def load_trans_data(self):
        return load_data()

    def embed_data(self, data, embedders: List):
        cat_data = data[:, :2].long()
        cont_data = data[:, 2:]

        embedder = embedders[0]

        pdg = cat_data[:, 0]
        prtc_pdg = embedder(pdg).squeeze()

        prtc_stat_code = cat_data[:, 1].unsqueeze(dim=1).float()

        return torch.cat([prtc_pdg, prtc_stat_code, cont_data], dim=1)

    def create_autoenc(self) -> PrtclVAE:
        return PrtclVAE(
            emb_features=EMB_FEATURES,
            latent_size=self.PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

    def loss(self, input_x, output_x, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (Variable, Variable, Variable):
        mse_loss = MSELoss()(input_x, output_x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return mse_loss + kld_loss * 1e-3, mse_loss, kld_loss

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(self.BATCH_SIZE) +
              ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
              ', EPOCHS: ' + str(epochs) +
              ', PRTCL_LATENT_SPACE_SIZE: ' + str(self.PRTCL_LATENT_SPACE_SIZE)
              )

        autoenc = self.create_autoenc()
        autoenc_optimizer = torch.optim.Adam(autoenc.parameters(), lr=self.LR)

        embedder = self.create_embedder()
        deembedder = self.create_deembedder()

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

        _all_data = self.load_trans_data()
        data_train, data_valid = self.prep_data(_all_data, batch_size=self.BATCH_SIZE, valid=0.1)

        particles = torch.tensor(particle_idxs(), device=self.device)
        particles.requires_grad = False

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):
                autoenc_optimizer.zero_grad()

                real_data: torch.Tensor = batch.to(self.device).detach()

                emb_data = self.embed_data(real_data, [embedder])

                gen_data, lat_mean, lat_logvar, lat_vec = autoenc(emb_data)

                loss, mse_loss, kld_loss = self.loss(
                    input_x=emb_data,
                    output_x=gen_data,
                    lat_mean=lat_mean,
                    lat_logvar=lat_logvar)

                loss.backward()
                autoenc_optimizer.step()

                self.train_deembeders(
                    tuples=[
                        (particles, embedder, deembedder)
                    ], epochs=2)

                if n_batch % 100 == 0:
                    self.print_deemb_quality(particles, embedder, deembedder)

                    self.show_heatmaps(emb_data[:30, :], gen_data[:30, :], reprod=False, save=True, epoch=epoch, batch=n_batch)
                    err_kld, err_wass = self.gen_show_comp_hists(
                        autoenc,
                        _all_data,
                        attr_idxs=[FEATURES - 8, FEATURES - 7, FEATURES - 6, FEATURES - 5],
                        embedders=[embedder],
                        emb=False,
                        deembedder=deembedder,

                        save=True,
                        epoch=epoch,
                        batch=n_batch
                    )

                    self.errs_kld.append(err_kld)
                    self.errs_wass.append(err_wass)

                    valid_loss = self._valid_loss(autoenc, embedder, data_valid)

                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'train loss: {"{:.6f}".format(round(loss.item(), 6))} :: '
                        f'kld loss: {"{:.6f}".format(round(kld_loss.item(), 6))} :: '
                        f'mse loss: {"{:.6f}".format(round(mse_loss.item(), 6))} :: '
                        f'valid loss: {"{:.6f}".format(round(valid_loss, 6))} :: '
                        f'err kld: {"{:.6f}".format(round(err_kld, 6))} :: '
                        f'err wass: {"{:.6f}".format(round(err_wass, 6))}'
                    )

            self._save_models(autoenc, embedder, deembedder)

            with open(self.ERRS_SAVE_PATH, 'wb') as handle:
                pickle.dump((self.errs_kld, self.errs_wass), handle, protocol=pickle.HIGHEST_PROTOCOL)

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
            emb_data = self.embed_data(batch_data.to(self.device), [embedder])
            out_prtcl_emb, lat_mean, lat_vec, lat_vec = autoenc(emb_data)
            train_loss = criterion(out_prtcl_emb, emb_data)

            loss += train_loss.item()

        loss /= len(valid_data_loader)

        return loss

    def gen_autoenc_data(self, sample_cnt, autoenc):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, self.PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = autoenc.decode(rand_input).detach()
        return generated_data
