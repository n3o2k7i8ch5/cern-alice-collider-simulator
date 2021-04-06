import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import MSELoss

from common.consts import EMB_FEATURES, PDG_EMB_DIM, parent_path, PDG_EMB_CNT, PARTICLE_DIM, \
    FEATURES

from i_trainer.i_trainer import ITrainer
from i_trainer.load_data import load_data, particle_idxs
from single_prtcl_generator_gan_pytorch.models.prtcl_gan_discriminator import PrtclGANDiscriminator
from single_prtcl_generator_gan_pytorch.models.prtcl_gan_generator import PrtclGANGenerator
from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder
from single_prtcl_generator_vae_pytorch.models.pdg_embedder import PDGEmbedder


class Trainer(ITrainer):
    BATCH_SIZE = 128
    PRTCL_LATENT_SPACE_SIZE = 20

    pdg_emb_cnt = PDG_EMB_CNT
    pdg_emb_dim = PDG_EMB_DIM

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

    GENERATOR_SAVE_PATH = parent_path() + 'data/single_prtc_gan_generator.model'
    DISCRIMINATOR_SAVE_PATH = parent_path() + 'data/single_prtc_gan_discriminator.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'
    PDG_EMBED_SAVE_PATH = parent_path() + 'data/pdg_embed.model'

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

    def train(self, epochs, load=False):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(self.BATCH_SIZE) +
              ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
              ', EPOCHS: ' + str(epochs) +
              ', PRTCL_LATENT_SPACE_SIZE: ' + str(self.PRTCL_LATENT_SPACE_SIZE)
              )

        generator, discriminator = self.create_model()
        gen_optim = torch.optim.Adam(generator.parameters(), lr=4e-5, betas=(0, .9))
        dis_optim = torch.optim.Adam(discriminator.parameters(), lr=4e-5, betas=(0, .9))

        embedder = PDGEmbedder(num_embeddings=self.pdg_emb_cnt, embedding_dim=self.pdg_emb_dim, device=self.device)
        deembedder: PDGDeembedder = PDGDeembedder(PDG_EMB_DIM, PDG_EMB_CNT, self.device)

        if load:
            print('LOADING MODEL STATES...')
            generator.load_state_dict(torch.load(Trainer.GENERATOR_SAVE_PATH))
            discriminator.load_state_dict(torch.load(Trainer.DISCRIMINATOR_SAVE_PATH))
            embedder.load_state_dict(torch.load(Trainer.PDG_EMBED_SAVE_PATH))
            deembedder.load_state_dict(torch.load(Trainer.PDG_DEEMBED_SAVE_PATH))

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

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):

                real_data: torch.Tensor = batch.to(self.device)
                emb_data = self.embed_data(real_data, [embedder]).detach()

                batch_size = len(batch)

                valid = torch.ones(batch_size, device=self.device, requires_grad=False)
                fake = torch.zeros(batch_size, device=self.device, requires_grad=False)

                # ======== Train Generator ======== #
                gen_optim.zero_grad()

                # Sample noise as generator input
                lat_fake = torch.randn(batch_size, self.PRTCL_LATENT_SPACE_SIZE, device=self.device)
                lat_fake = Variable(
                    torch.tensor(
                        np.random.normal(0, 1, (batch_size, self.PRTCL_LATENT_SPACE_SIZE)),
                        device=self.device
                    ).float()
                )
                # Generate a batch of images
                gen_data = generator(lat_fake)

                # Loss measures generator's ability to fool the discriminator
                g_loss = MSELoss()(discriminator(gen_data), valid)

                g_loss.backward()
                gen_optim.step()

                # ======== Train Discriminator ======== #
                dis_optim.zero_grad()

                real_loss = MSELoss()(discriminator(emb_data), valid)
                fake_loss = MSELoss()(discriminator(gen_data.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                dis_optim.step()

                # self.train_deembeder(deembedder=deembedder, embedder=embedder, epochs=1, device=self.device)

                if n_batch % 500 == 0:
                    self.print_deemb_quality(torch.tensor(particle_idxs(), device=self.device), embedder, deembedder)

                    #show_quality(emb_data, gen_data, feature_range=self.show_feat_rng, save=True)
                    self.show_heatmaps(emb_data[:30, :], gen_data[:30, :], reprod=False, save=True, epoch=epoch, batch=n_batch)
                    self.gen_show_comp_hists(
                        generator,
                        _data,
                        attr_idxs=[FEATURES-8, FEATURES-7, FEATURES-6, FEATURES-5],
                        embedders=[embedder],
                        emb=False,
                        deembedder=deembedder,

                        save=True,
                        epoch=epoch,
                        batch=n_batch
                    )

                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'generator loss: {"{:.6f}".format(round(g_loss.item(), 6))} :: '
                        f'discriminator loss: {"{:.6f}".format(round(d_loss.item(), 6))}'
                    )

            self._save_models(generator, discriminator, embedder, deembedder)

        return generator, discriminator, embedder, deembedder

    def _save_models(self, generator, discriminator, embedder, deembedder):
        print('Saving generator model')
        torch.save(generator.state_dict(), Trainer.GENERATOR_SAVE_PATH)
        print('Saving discriminator model')
        torch.save(discriminator.state_dict(), Trainer.DISCRIMINATOR_SAVE_PATH)
        print('Saving embed model')
        torch.save(embedder.state_dict(), Trainer.PDG_EMBED_SAVE_PATH)
        print('Saving deembedder model')
        torch.save(deembedder.state_dict(), Trainer.PDG_DEEMBED_SAVE_PATH)

    def gen_autoenc_data(self, sample_cnt, generator):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, self.PRTCL_LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = generator(rand_input).detach()
        return generated_data
