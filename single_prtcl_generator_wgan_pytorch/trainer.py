import pickle

import numpy as np
import torch
from torch import autograd
from torch.optim import Adam

from common.consts import EMB_FEATURES, parent_path, PARTICLE_DIM, FEATURES

from i_trainer.i_trainer import ITrainer
from i_trainer.load_data import load_data, particle_idxs
from single_prtcl_generator_gan_pytorch.models.prtcl_gan_discriminator import PrtclGANDiscriminator
from single_prtcl_generator_gan_pytorch.models.prtcl_gan_generator import PrtclGANGenerator


class Trainer(ITrainer):
    BATCH_SIZE = 128*8
    CRITIC_ITERATIONS = 2#10
    PRTCL_LATENT_SPACE_SIZE = 12
    LR = 5e-5

    errs_kld = []
    errs_wass = []

    GENERATOR_SAVE_PATH = parent_path() + 'data/single_prtc_wgan_generator.model'
    CRITIC_SAVE_PATH = parent_path() + 'data/single_prtc_wgan_critic.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'
    ERRS_SAVE_PATH = parent_path() + 'data/errs_wgan.model'

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

    def create_model(self) -> (PrtclGANGenerator, PrtclGANDiscriminator):
        generator = PrtclGANGenerator(
            emb_features=EMB_FEATURES,
            latent_size=self.PRTCL_LATENT_SPACE_SIZE,
            device=self.device
        )

        discriminator = PrtclGANDiscriminator(
            emb_features=EMB_FEATURES,
            device=self.device
        )

        return generator, discriminator

    def calc_gradient_penalty(self, netD, real_data, fake_data, batch_size):

        LAMBDA = 1

        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(self.device)

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(self.BATCH_SIZE) +
              ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
              ', EPOCHS: ' + str(epochs) +
              ', PRTCL_LATENT_SPACE_SIZE: ' + str(self.PRTCL_LATENT_SPACE_SIZE)
              )

        generator, discriminator = self.create_model()
        gen_optim = Adam(generator.parameters(), lr=self.LR, betas=(.1, .9))
        dis_optim = Adam(discriminator.parameters(), lr=self.LR, betas=(.1, .9))

        embedder = self.create_embedder()
        deembedder = self.create_deembedder()

        if load:
            print('LOADING MODEL STATES...')
            try:
                generator.load_state_dict(torch.load(Trainer.GENERATOR_SAVE_PATH))
            except Exception:
                print('Problem loading generator!')
                pass

            try:
                discriminator.load_state_dict(torch.load(Trainer.CRITIC_SAVE_PATH))
            except Exception:
                print('Problem loading critic!')
                pass

            try:
                embedder.load_state_dict(torch.load(Trainer.PDG_EMBED_SAVE_PATH))
            except Exception:
                print('Problem loading embeder!')
                pass

            try:
                deembedder.load_state_dict(torch.load(Trainer.PDG_DEEMBED_SAVE_PATH))
            except Exception:
                print('Problem loading deembeder!')
                pass

        print('GENERATOR')
        print(generator)
        print('DISCRIMINATOR')
        print(discriminator)
        print('EMBEDDER')
        print(embedder)
        print('DEEMBEDDER')
        print(deembedder)

        _data = load_data()
        data_train, data_valid = self.prep_data(_data, batch_size=self.BATCH_SIZE, valid=0.1)

        particles = torch.tensor(particle_idxs(), device=self.device)
        particles.requires_grad = False

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):

                real_data: torch.Tensor = batch.to(self.device)
                emb_data = self.embed_data(real_data, [embedder]).detach()

                batch_size = len(batch)

                # ======== Train Generator ======== #

                gen_optim.zero_grad()

                for p in discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

                # Sample noise as generator input
                lat_fake = torch.randn(batch_size, self.PRTCL_LATENT_SPACE_SIZE, device=self.device)
                gen_data = generator(lat_fake)

                output = discriminator(gen_data)
                gen_loss = -torch.mean(output)

                gen_loss.backward()
                gen_optim.step()

                for p in discriminator.parameters():
                    p.requires_grad = True  # to avoid computation

                # ======== Train Discriminator ======== #
                for _ in range(self.CRITIC_ITERATIONS):
                    critic_real = discriminator(emb_data)
                    critic_fake = discriminator(gen_data.detach())
                    gp = self.calc_gradient_penalty(discriminator, emb_data, gen_data, batch_size)
                    critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp

                    critic_loss.backward()
                    dis_optim.step()

                    dis_optim.zero_grad()

                self.train_deembeders([
                        (particles, embedder, deembedder),
                    ], epochs=2)

                if n_batch % 100 == 0:
                    self.print_deemb_quality(particles, embedder, deembedder)

                    self.show_heatmaps(emb_data[:30, :], gen_data[:30, :], reprod=False, save=True, epoch=epoch, batch=n_batch)
                    err_kld, err_wass = self.gen_show_comp_hists(
                        generator,
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

                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'generator loss: {"{:.6f}".format(round(gen_loss.item(), 6))} :: '
                        f'critic loss: {"{:.6f}".format(round(critic_loss.item(), 6))} :: '
                        f'err kld: {"{:.6f}".format(round(err_kld, 6))} :: '
                        f'err wass: {"{:.6f}".format(round(err_wass, 6))}'
                    )

            self._save_models(generator, discriminator, embedder, deembedder)

            with open(self.ERRS_SAVE_PATH, 'wb') as handle:
                pickle.dump((self.errs_kld, self.errs_wass), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return generator, discriminator, embedder, deembedder

    def _save_models(self, generator, discriminator, embedder, deembedder):
        print('Saving generator model')
        torch.save(generator.state_dict(), Trainer.GENERATOR_SAVE_PATH)
        print('Saving critic model')
        torch.save(discriminator.state_dict(), Trainer.CRITIC_SAVE_PATH)
        print('Saving embed model')
        torch.save(embedder.state_dict(), Trainer.PDG_EMBED_SAVE_PATH)
        print('Saving deembedder model')
        torch.save(deembedder.state_dict(), Trainer.PDG_DEEMBED_SAVE_PATH)

    def gen_autoenc_data(self, sample_cnt, generator):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, self.PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = generator(rand_input).detach()
        return generated_data
