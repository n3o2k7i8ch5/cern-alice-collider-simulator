import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import numpy as np

from common.consts import PRTCL_LATENT_SPACE_SIZE, EMB_FEATURES, PDG_EMB_DIM, parent_path, PDG_EMB_CNT, BATCH_SIZE, \
    PARTICLE_DIM
from common.prtcl_vae import PrtclVAE

from common.show_quality import show_lat_histograms, show_quality
from i_trainer.i_trainer import ITrainer
from i_trainer.load_data import load_data
from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder


class Trainer(ITrainer):

    pdg_emb_cnt = PDG_EMB_CNT
    pdg_emb_dim = PDG_EMB_DIM
    show_feat_rng = -7, -4

    def load_data(self):
        return

    AUTOENC_SAVE_PATH = parent_path() + 'data/single_prtc_autoenc.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'

    def loss(self, input_x, output_x, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (
    Variable, Variable, Variable):
        mse_loss = MSELoss()(input_x, output_x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return mse_loss + kld_loss * 1e-4, mse_loss, kld_loss

    def create_model(self) -> PrtclVAE:
        return PrtclVAE(
            emb_features=EMB_FEATURES,
            latent_size=PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(BATCH_SIZE) +
              ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
              ', EPOCHS: ' + str(epochs) +
              ', PRTCL_LATENT_SPACE_SIZE: ' + str(PRTCL_LATENT_SPACE_SIZE)
              )

        autoenc = self.create_model()
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

        _data = load_data()
        data_train, data_valid = self.prep_data(_data, batch_size=BATCH_SIZE, valid=0.1)

        uniq_cred_idxs = torch.tensor(self.par, device=self.device)

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

                self.train_deembeder(deembedder=deembedder, embedder=embedder, epochs=1, device=self.device)

                if n_batch % 500 == 0:
                    show_lat_histograms(lat_mean=lat_mean, lat_logvar=lat_logvar)
                    self.show_deemb_quality(embedder, deembedder)
                    valid_loss = self._valid_loss(autoenc, embedder, data_valid)

                    show_quality(emb_data, gen_data, feature_range=self.show_feat_rng, save=True)
                    self.show_img_comparison(emb_data[:30, :], gen_data[:30, :])

                    self.show_real_gen_data_comparison(autoenc, real_data, [embedder], emb=False, deembedder=deembedder, save=True)
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

    def gen_autoenc_data(self, sample_cnt, autoenc):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = autoenc.decode(rand_input).detach()
        return generated_data
