import pickle

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import MSELoss

from _legacy.common.consts import EMB_FEATURES, parent_path, PARTICLE_DIM, FEATURES

from _legacy.i_trainer.i_trainer import ITrainer
from _legacy.i_trainer.load_data import load_data, particle_idxs
from _legacy.single_prtcl_generator_wae_pytorch.models.prtcl_wae_decoder import PrtclWAEDecoder
from _legacy.single_prtcl_generator_wae_pytorch.models.prtcl_wae_discriminator import PrtclWAEDiscriminator
from _legacy.single_prtcl_generator_wae_pytorch.models.prtcl_wae_encoder import PrtclWAEEncoder


class Trainer(ITrainer):
    BATCH_SIZE = 128*8

    PRTCL_LATENT_SPACE_SIZE = 12
    LR = 5e-5

    errs_kld = []
    errs_wass = []

    ENCODER_SAVE_PATH = parent_path() + 'data/single_prtc_wae_encoder.model'
    DECODER_SAVE_PATH = parent_path() + 'data/single_prtc_wae_decoder.model'
    DISCRIMINATOR_SAVE_PATH = parent_path() + 'data/single_prtc_wae_discriminator.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'
    ERRS_SAVE_PATH = parent_path() + 'data/errs_gan.model'

    def load_trans_data(self):
        return load_data()

    def embed_data(self, data, embedders):
        cat_data = data[:, :2].long()
        cont_data = data[:, 2:]

        embedder = embedders[0]

        pdg = cat_data[:, 0]
        prtc_pdg = embedder(pdg).squeeze()

        prtc_stat_code = cat_data[:, 1].unsqueeze(dim=1).float()

        return torch.cat([prtc_pdg, prtc_stat_code, cont_data], dim=1)

    def loss(self, input_x, output_x) -> (Variable):
        return MSELoss()(input_x, output_x)

    def create_model(self) -> (PrtclWAEEncoder, PrtclWAEDecoder, PrtclWAEDiscriminator):
        encoder = PrtclWAEEncoder(
            emb_features=EMB_FEATURES,
            latent_size=self.PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

        decoder = PrtclWAEDecoder(
            emb_features=EMB_FEATURES,
            latent_size=self.PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

        discriminator = PrtclWAEDiscriminator(
            latent_size=self.PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

        return encoder, decoder, discriminator

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(Trainer.BATCH_SIZE) +
              ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
              ', EPOCHS: ' + str(epochs) +
              ', PRTCL_LATENT_SPACE_SIZE: ' + str(self.PRTCL_LATENT_SPACE_SIZE)
              )

        encoder, decoder, discriminator = self.create_model()
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=self.LR)
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=self.LR)
        dis_optim = torch.optim.Adam(discriminator.parameters(), lr=self.LR)

        embedder = self.create_embedder()
        deembedder = self.create_deembedder()

        if load:
            print('LOADING MODEL STATES...')
            encoder.load_state_dict(torch.load(Trainer.ENCODER_SAVE_PATH))
            decoder.load_state_dict(torch.load(Trainer.DECODER_SAVE_PATH))
            discriminator.load_state_dict(torch.load(Trainer.DISCRIMINATOR_SAVE_PATH))
            embedder.load_state_dict(torch.load(Trainer.PDG_EMBED_SAVE_PATH))
            deembedder.load_state_dict(torch.load(Trainer.PDG_DEEMBED_SAVE_PATH))

        print('AUTOENCODER')
        print(encoder)
        print(decoder)
        print(discriminator)
        print('EMBEDDER')
        print(embedder)
        print('DEEMBEDDER')
        print(deembedder)

        _data = load_data()
        data_train, data_valid = self.prep_data(_data, batch_size=Trainer.BATCH_SIZE, valid=0.1)

        particles = torch.tensor(particle_idxs(), device=self.device)
        particles.requires_grad = False

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):
                encoder.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

                real_data: torch.Tensor = batch.to(self.device)
                emb_data = self.embed_data(real_data, [embedder]).detach()

                batch_size = len(batch)

                zeros = torch.zeros(batch_size, device=self.device, requires_grad=False)
                ones = torch.ones(batch_size, device=self.device, requires_grad=False)

                # ======== Train Discriminator ======== #
                decoder.freeze(True)
                encoder.freeze(True)
                discriminator.freeze(False)

                lat_fake = torch.randn(batch_size, self.PRTCL_LATENT_SPACE_SIZE, device=self.device)
                disc_fake = discriminator(lat_fake)

                lat_real = encoder(emb_data)
                disc_real = discriminator(lat_real)

                loss_fake = MSELoss()(disc_fake, zeros)
                loss_real = MSELoss()(disc_real, ones)

                loss_fake.backward()
                loss_real.backward()

                dis_optim.step()

                # ======== Train Generator ======== #
                decoder.freeze(False)
                encoder.freeze(False)
                discriminator.freeze(True)

                lat_real = encoder(emb_data)
                recon_data = decoder(lat_real)
                d_real = discriminator(encoder(emb_data))

                recon_loss = MSELoss()(emb_data, recon_data)
                d_loss = MSELoss()(d_real, zeros)

                recon_loss.backward()
                d_loss.backward()

                enc_optim.step()
                dec_optim.step()

                self.train_deembeders([
                    (particles, embedder, deembedder),
                ], epochs=2)

                if n_batch % 100 == 0:
                    self.print_deemb_quality(particles, embedder, deembedder)

                    self.show_heatmaps(emb_data[:30, :], recon_data[:30, :], reprod=False, save=True, epoch=epoch, batch=n_batch)
                    err_kld, err_wass = self.gen_show_comp_hists(
                        decoder,
                        _data,
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

                    valid_loss = self._valid_loss(encoder, decoder, embedder, data_valid)

                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'train loss: {"{:.6f}".format(round(recon_loss.item(), 6))} :: '
                        f'valid loss: {"{:.6f}".format(round(valid_loss, 6))} :: '
                        f'err kld: {"{:.6f}".format(round(err_kld, 6))} :: '
                        f'err wass: {"{:.6f}".format(round(err_wass, 6))}'
                    )

            self._save_models(encoder, decoder, discriminator, embedder, deembedder)

            with open(self.ERRS_SAVE_PATH, 'wb') as handle:
                pickle.dump((self.errs_kld, self.errs_wass), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return encoder, decoder, discriminator, embedder, deembedder

    def _save_models(self, encoder, decoder, discriminator, embedder, deembedder):
        print('Saving encoder model')
        torch.save(encoder.state_dict(), Trainer.ENCODER_SAVE_PATH)
        print('Saving decoder model')
        torch.save(decoder.state_dict(), Trainer.DECODER_SAVE_PATH)
        print('Saving discriminator model')
        torch.save(discriminator.state_dict(), Trainer.DISCRIMINATOR_SAVE_PATH)
        print('Saving embed model')
        torch.save(embedder.state_dict(), Trainer.PDG_EMBED_SAVE_PATH)
        print('Saving deembedder model')
        torch.save(deembedder.state_dict(), Trainer.PDG_DEEMBED_SAVE_PATH)

    def _valid_loss(self, encoder, decoder, embedder, valid_data_loader) -> float:
        loss = 0
        criterion = MSELoss()

        for batch_data in valid_data_loader:
            emb_data = self.embed_data(batch_data.to(self.device), [embedder])
            recon_data = decoder(encoder(emb_data))
            train_loss = criterion(recon_data, emb_data)

            loss += train_loss.item()

        loss /= len(valid_data_loader)

        return loss

    def gen_autoenc_data(self, sample_cnt, decoder):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, self.PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = decoder(rand_input).detach()
        return generated_data
