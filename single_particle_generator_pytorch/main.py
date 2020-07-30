import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import matplotlib.pyplot as plt

from common.consts import *
from common.models_prtc_embed_autoenc import Autoencoder
from single_particle_generator_pytorch.load_data import load_data

# device = get_device()

# torch.cuda.set_device(0)
device = torch.device("cuda")

data = load_data()

data_train = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


def noise(size, device) -> torch.Tensor:
    return torch.randn(size, INP_RAND_SIZE).to(device=device)


def _loss(input_x, output_x, lat_mu: torch.Tensor, lat_var: torch.Tensor) -> Variable:
    mse_loss = MSELoss()(input_x, output_x)

    # kl_loss = 0.5 * torch.sum(lat_var.exp() + lat_mu.pow(2) - 1.0 - lat_var) / len(lat_var.flatten())

    kl_loss = MSELoss()(lat_mu, torch.zeros(size=lat_mu.size()).to(device)) + MSELoss()(lat_var, torch.ones(
        size=lat_var.size()).to(device))

    # print('mse_loss ' + str(mse_loss.item()))
    # print('kl_loss ' + str(kl_loss.item()))

    return mse_loss + kl_loss / 500


### TRAINING


def show_quality(real_data, fake_data):
    img_smpls_to_shw = 80

    real_data = real_data.cpu()
    fake_data = fake_data.cpu()

    print('REAL DATA')
    for i in range(0, real_data.shape[0], 10000):
        plt.figure('Real img:' + str(i))
        plt.imshow(real_data.detach().split(split_size=img_smpls_to_shw, dim=0)[0])
        # plt.colorbar()
        plt.ion()
        plt.figure('Real hist:' + str(i))
        plt.hist(real_data.detach().flatten(), 100)
        plt.show()
    plt.pause(0.001)

    print('FAKE DATA')
    for i in range(0, fake_data.shape[0], 10000):
        plt.figure('Fake img:' + str(i))
        plt.imshow(fake_data.detach().split(split_size=img_smpls_to_shw, dim=0)[0])
        # plt.colorbar()
        plt.ion()
        plt.figure('Fake hist:' + str(i))
        plt.hist(fake_data.detach().flatten(), 100)
        plt.show()

    plt.pause(0.001)

    real_data.to(device)
    fake_data.to(device)


print('AUTOENCODER')
LATENT_SPACE_SIZE = 6
_autoenc_in, _autoenc_out = Autoencoder.create(emb_features=EMB_FEATURES, latent=LATENT_SPACE_SIZE, device=device)
autoenc = Autoencoder(_autoenc_in, _autoenc_out)
# summary(autoenc, input_size=(PADDING, FEATURES))

autoenc_optimizer = optim.Adam(autoenc.parameters(), lr=0.001)


def train_autoenc():
    for epoch in range(1):

        err: torch.Tensor = torch.Tensor()
        emb_data: torch.Tensor = torch.Tensor()
        gen_data: torch.Tensor = torch.Tensor()

        for n_batch, batch in enumerate(data_train):
            autoenc_optimizer.zero_grad()
            real_data: torch.Tensor = batch.to(device=device)

            real_data = real_data.detach()

            emb_data, gen_data, lat_mu, lat_var = autoenc(real_data)

            err = _loss(emb_data, gen_data, lat_mu, lat_var)
            err.backward()
            autoenc_optimizer.step()

            if n_batch % 5000 == 0:
                print('' + str(n_batch) + '/' + str(len(data_train)))
                show_quality(emb_data, gen_data)

                plt.figure('Lattent mu')
                plt.hist(lat_mu.cpu().detach().flatten(), 100)

                plt.figure('Lattent var')
                plt.hist(lat_var.cpu().detach().flatten(), 100)
                plt.show()

                plt.pause(0.001)

        print(err.item())
        show_quality(emb_data, gen_data)

    torch.save(autoenc.state_dict(), parent_path() + 'data/single_prtc_autoenc_model')


def gen_autoenc_data(autoenc_out):
    np_input = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE, LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    generated_data = autoenc_out.forward(rand_input).detach()
    return generated_data


# GENERATE DATA
# GENERATE DATA
# '''
autoenc.load_state_dict(torch.load(parent_path() + 'data/single_prtc_autoenc_model'))

emb_data: torch.Tensor = torch.Tensor()
gen_data: torch.Tensor = torch.Tensor()

for n_batch, batch in enumerate(data_train):
    real_data = batch.to(device=device)

    real_data = real_data.detach()

    _emb_data, gen_data, lat_mu, lat_var = autoenc(real_data)

    if n_batch == 0:
        emb_data = _emb_data.cpu()
    else:
        emb_data = torch.cat([_emb_data.cpu(), emb_data], dim=0)

    if n_batch == 500:
        break

gen_data_out: torch.Tensor = torch.Tensor().cpu()
for i in range(500):
    if i == 0:
        gen_data_out = gen_autoenc_data(autoenc.auto_out).cpu()
    else:
        gen_data_out = torch.cat([gen_autoenc_data(autoenc.auto_out).cpu(), gen_data_out], dim=0)

show_quality(emb_data, gen_data_out)

import time

time.sleep(1000)
# '''

# /GENERATE DATA
# /GENERATE DATA

train_autoenc()
