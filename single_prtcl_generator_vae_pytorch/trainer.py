from typing import List

import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import numpy as np

from common.consts import PRTCL_LATENT_SPACE_SIZE, EMB_FEATURES, PDG_EMB_DIM, parent_path, PDG_EMB_CNT, PARTICLE_DIM, \
    particle_idxs, FEATURES
from common.models.pdg_embedder import PDGEmbedder
from common.prtcl_vae import PrtclVAE
from common.show_quality import show_lat_histograms, show_quality
from i_trainer.i_trainer import ITrainer
from i_trainer.load_data import load_data
from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder


class Trainer(ITrainer):

    BATCH_SIZE = 512

    pdg_emb_cnt = PDG_EMB_CNT
    pdg_emb_dim = PDG_EMB_DIM
    show_feat_rng = -7, -4

    AUTOENC_SAVE_PATH = parent_path() + 'data/single_prtc_autoenc.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'

    def load_trans_data(self):
        return load_data()

    def loss(self, input_x, output_x, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (
    Variable, Variable, Variable):
        mse_loss = MSELoss()(input_x, output_x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return mse_loss + kld_loss * 1e-3, mse_loss, kld_loss

    def create_autoenc(self) -> PrtclVAE:
        return PrtclVAE(
            emb_features=EMB_FEATURES,
            latent_size=PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

    def create_embedder(self):
        return PDGEmbedder(num_embeddings=self.pdg_emb_cnt, embedding_dim=self.pdg_emb_dim, device=self.device)

    def create_deembedder(self) -> PDGDeembedder:
        return PDGDeembedder(self.pdg_emb_dim, self.pdg_emb_cnt, self.device)

    def create_train_deembedder(self, embedder, epochs):
        deembedder = self.create_deembedder()
        self.train_deembeders(
            tuples=[(torch.tensor(particle_idxs(), device=self.device), embedder, deembedder)],
            epochs=epochs,
        )
        return deembedder

    def embed_data(self, data, embedders: List):
        cat_data = data[:, :2].long()
        cont_data = data[:, 2:]

        embedder = embedders[0]

        pdg = cat_data[:, 0]
        prtc_pdg = embedder(pdg).squeeze()

        prtc_stat_code = cat_data[:, 1].unsqueeze(dim=1).float()

        return torch.cat([prtc_pdg, prtc_stat_code, cont_data], dim=1)

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(self.BATCH_SIZE) +
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

        _all_data = self.load_trans_data()
        data_train, data_valid = self.prep_data(_all_data, batch_size=self.BATCH_SIZE, valid=0.1)

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
                    tuples=[(
                        torch.tensor(particle_idxs(), device=self.device),
                        embedder,
                        deembedder
                    )],
                    epochs=1,
                )

                if n_batch % 500 == 0:
                    self.show_heatmaps(
                        emb_data[:30, :],
                        gen_data[:30, :],
                        reprod=False,
                        save=True,
                        epoch=epoch,
                        batch=n_batch
                    )

                    self.gen_show_comp_hists(
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

                    '''
                    show_lat_histograms(lat_mean=lat_mean, lat_logvar=lat_logvar)
                    self.print_deemb_quality(
                        torch.tensor(particle_idxs(), device=self.device),
                        embedder,
                        deembedder
                    )
                    '''
                    valid_loss = self._valid_loss(autoenc, embedder, data_valid)

                    #show_quality(emb_data, gen_data, feature_range=self.show_feat_rng, save=True)
                    #self.show_heatmaps(emb_data[:30, :], gen_data[:30, :])

                    #self.gen_show_comp_hists(autoenc, _all_data, [embedder], emb=False, deembedder=deembedder, save=True)
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
            emb_data = self.embed_data(batch_data.to(self.device), [embedder])
            out_prtcl_emb, lat_mean, lat_vec, lat_vec = autoenc(emb_data)
            train_loss = criterion(out_prtcl_emb, emb_data)

            loss += train_loss.item()

        loss /= len(valid_data_loader)

        return loss

    def gen_autoenc_data(self, sample_cnt, autoenc):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = autoenc.decode(rand_input).detach()
        return generated_data
