import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as func

import matplotlib.pyplot as plt
from torchsummary import summary

from common.consts import *
from common.load_event_data import load_data
from common.model_pdg_emb_deemb import PDGDeembedder, show_deemb_quality
from common.show_plots import show_plots
from event_generator_pytorch.models_embedding_autoencoder import Autoencoder

# device = get_device()

# torch.cuda.set_device(0)
device = torch.device("cuda")
data_cat, data_cont, max_length = load_data()

padded_data_cat: torch.Tensor = pad_sequence(data_cat, batch_first=True).split(split_size=PADDING, dim=1)[0]
padded_data_cont: torch.Tensor = pad_sequence(data_cont, batch_first=True).split(split_size=PADDING, dim=1)[0]

data_train = DataLoader(
    TensorDataset(padded_data_cat.float(), padded_data_cont),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_deemb_train = DataLoader(
    torch.tensor(particle_idxs()).to(device),
    batch_size=len(particle_idxs()),
    shuffle=True
)


def noise(size, device) -> torch.Tensor:
    # Generates a 1-d vector of gaussian sampled random values
    return torch.randn(size, INP_RAND_SIZE).to(device=device)


def _loss(input_x, output_x, lat_mu, lat_var, real_one_hot, gen_one_hot, show_partial=False) -> Variable:
    mse_loss = MSELoss()(input_x, output_x)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_var - lat_mu ** 2 - lat_var.exp(), dim=1), dim=0)

    # KL divergence policzyć jak nalezy
    # dane syntetyczne

    deemb_loss = MSELoss()(real_one_hot, gen_one_hot)

    # return deemb_loss

    if show_partial:
        print('\tmse_loss ' + str(mse_loss.item()))
        print('\tkld_loss ' + str(kld_loss.item()))
        print('\tdeemb_loss ' + str(deemb_loss.item()))

    return mse_loss + kld_loss*0.0001 + deemb_loss


### TRAINING


print('AUTOENCODER')
LATENT_SPACE_SIZE = 560  # 48
autoenc_in, autoenc_out = Autoencoder.create(
    samples=PADDING, emb_features=EMB_FEATURES, latent=LATENT_SPACE_SIZE, device=device)

autoenc = Autoencoder(autoenc_in, autoenc_out)
deembeder: PDGDeembedder = PDGDeembedder(PDG_EMB_DIM, PDG_EMB_CNT, device)

summary(autoenc_in, input_size=(PADDING, FEATURES))  # (PADDING, FEATURES))
summary(autoenc_out, input_size=(LATENT_SPACE_SIZE,))
summary(deembeder, input_size=(PDG_EMB_DIM,))

# Z jakiegoś arcyciekawego powodu Adam działa świetnie, a SGD prawie w ogóle.
# Chodzi głównie o Deembeder, który na SGD się nie chce uczyć.
autoenc_optimizer = optim.Adam(params=autoenc.parameters(), lr=0.0001)
deembed_optimizer = optim.Adam(params=deembeder.parameters(), lr=0.0001)

EPOCHS = 15


def train_autoenc():
    print('TRAINING MODEL:'
          ' BATCH_SIZE = ' + str(BATCH_SIZE) +
          ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
          ', EPOCHS: ' + str(EPOCHS) +
          ', LATENT_SPACE_SIZE: ' + str(LATENT_SPACE_SIZE)
          )

    for epoch in range(EPOCHS):

        err: torch.Tensor = torch.Tensor()
        emb_data: torch.Tensor = torch.Tensor()
        gen_data: torch.Tensor = torch.Tensor()
        lat_mu: torch.Tensor = torch.Tensor()
        lat_var: torch.Tensor = torch.Tensor()
        real_one_hot = func.one_hot(torch.tensor(particle_idxs(), device=device), num_classes=PDG_EMB_CNT).float()
        gen_one_hot: torch.Tensor = torch.Tensor()

        for n_batch, (batch_cat, batch_cont) in enumerate(data_train):
            autoenc_optimizer.zero_grad()
            deembed_optimizer.zero_grad()

            # TRAIN V-AUTOENCODER
            real_data_cat: torch.Tensor = batch_cat.to(device=device)
            real_data_cont: torch.Tensor = batch_cont.to(device=device)

            real_data = torch.cat([real_data_cat, real_data_cont], dim=2).detach()

            emb_data, gen_data, lat_mu, lat_var = autoenc(real_data)

            # TRAIN DEEMBEDER
            embed = autoenc.auto_in.pdg_emb(real_one_hot)
            gen_one_hot = deembeder(embed)

            err = _loss(emb_data, gen_data, lat_mu, lat_var, real_one_hot, gen_one_hot, show_partial=n_batch % 100 == 0)
            err.backward()
            autoenc_optimizer.step()
            deembed_optimizer.step()

        show_deemb_quality(autoenc_in.pdg_emb, deembeder, device)
        #show_real_gen_data_comparison(load=False)

        print(err.item())

    torch.save(autoenc.state_dict(), parent_path() + 'data/event_autoenc_model')
    torch.save(deembeder.state_dict(), parent_path() + 'data/event_pdg_deembed_model')

    show_real_gen_data_comparison()


def generate_data(autoenc_out, deembeder: PDGDeembedder):
    np_input = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE, LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    gen_emb_data = autoenc_out.forward(rand_input).detach()

    gen_data = deembeder.deemb(gen_emb_data)

    return gen_data


def show_real_gen_data_comparison(load: bool = True):
    print('GENERATING DATA')

    if load:
        autoenc.load_state_dict(torch.load(parent_path() + 'data/event_autoenc_model'))
        deembeder.load_state_dict(torch.load(parent_path() + 'data/event_pdg_deembed_model'))

    for param in autoenc.parameters():
        param.requires_grad = False

    for param in deembeder.parameters():
        param.requires_grad = False

    real_data: torch.Tensor = torch.Tensor()
    gen_data: torch.Tensor = torch.Tensor()

    BATCHES_TO_DISP = 40

    for n_batch, (batch_cat, batch_cont) in enumerate(data_train):

        _real_data = torch.cat([batch_cat.cpu(), batch_cont.cpu()], dim=2)
        if n_batch == 0:
            real_data = _real_data.cpu()
        else:
            real_data = torch.cat([_real_data.cpu(), real_data], dim=0)

        if n_batch == BATCHES_TO_DISP:
            break

    gen_data_out: torch.Tensor = torch.Tensor().cpu()
    for i in range(BATCHES_TO_DISP):
        if i == 0:
            gen_data_out = generate_data(autoenc.auto_out, deembeder).cpu()
        else:
            gen_data_out = torch.cat([generate_data(autoenc.auto_out, deembeder).cpu(), gen_data_out], dim=0)

    show_plots(real_data, gen_data_out)

    for param in autoenc.parameters():
        param.requires_grad = True

    for param in deembeder.parameters():
        param.requires_grad = True


train_autoenc()

# show_real_gen_data_comparison()
