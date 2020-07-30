
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import numpy as np

from device import get_device
from load_data import load_data

from models_embed_params import *

import matplotlib.pyplot as plt

from models_embedding_autoencoder import Autoencoder, AutoencoderOut

'''

Step 1:
- Load data (event list)
    d - discrete (categorical)
    c - continous
    Event:
        [d d d ... c c c]
        [d d d ... c c c]
        ...
        [d d d ... c c c]
        [d d d ... c c c]

    Embed:
        Use pretrained value -> embed models.
        1. d -> one_hot         d -> [0 ... 0 1 0 0 ... 0]
        2. one_hot -> vec       [0 ... 0 1 0 0 ... 0] -> [v1, v2, v3, ... vn]

    Training.

    Generator: -> [vec, vec, vec ... c c c]

'''

device = get_device()

torch.cuda.set_device(0)
device = torch.device("cuda")

data_cat, data_cont, max_length = load_data()

padded_data_cat: torch.Tensor = \
    torch.nn.utils.rnn.pad_sequence(data_cat, batch_first=True).split(split_size=PADDING, dim=1)[0]
padded_data_cont: torch.Tensor = \
    torch.nn.utils.rnn.pad_sequence(data_cont, batch_first=True).split(split_size=PADDING, dim=1)[0]

data_train = DataLoader(
    TensorDataset(padded_data_cat, padded_data_cont),
    batch_size=BATCH_SIZE,
    shuffle=True
)


def noise(size, device) -> torch.Tensor:
    # Generates a 1-d vector of gaussian sampled random values
    return torch.randn(size, INP_RAND_SIZE).to(device=device)


param_emb_container = ParamEmbedContainer(device)
param_emb_container.learn(False)

summary(param_emb_container.autoenc_part_refer, input_size=(PADDING, PART_REFER_EMB_CNT))
summary(param_emb_container.autoenc_pdg, input_size=(PADDING, PDG_EMB_CNT))


def _loss(input_x, output_x, lat_mu, lat_var) -> Variable:
    mse_loss = MSELoss()(input_x, output_x)

    kl_loss = 0.5 * torch.sum(torch.exp(lat_var) + lat_mu.pow(2) - 1.0 - lat_var)

    print('mse_loss ' + str(mse_loss))
    print('kl_loss ' + str(kl_loss))

    return mse_loss + kl_loss


# loss = MSELoss()


### TRAINING


def show_quality(real_data, fake_data):
    img_smpls_to_shw = 80

    real_data = real_data.cpu()
    fake_data = fake_data.cpu()

    print('REAL DATA')
    for i in range(0, real_data.shape[0], 10000):
        real_data_deemb = param_emb_container.deembed(real_data).detach()

        plt.figure('Real img:' + str(i))
        plt.imshow(real_data_deemb[i, :, :].split(split_size=img_smpls_to_shw, dim=0)[0])
        # plt.colorbar()
        plt.ion()
        plt.show()
        plt.figure('Real hist:' + str(i))
        plt.hist(real_data_deemb[i, :, 5].cpu().flatten(), 100)
        plt.show()
    plt.pause(0.001)

    # plt.figure(1)
    # plt.hist(real_data[:, :, 6].detach().cpu().numpy().flatten(), 100)

    print('FAKE DATA')
    for i in range(0, fake_data.shape[0], 10000):
        fake_data_deemb = param_emb_container.deembed(fake_data)

        plt.figure('Fake img:' + str(i))
        plt.imshow(fake_data_deemb[i, :, :].split(split_size=img_smpls_to_shw, dim=0)[0].detach().cpu())
        # plt.colorbar()
        plt.ion()
        plt.show()
        plt.figure('Fake hist:' + str(i))
        plt.hist(fake_data_deemb[i, :, 5].detach().cpu().flatten(), 100)
        plt.show()

    plt.pause(0.001)

    # plt.figure(1)
    # plt.hist(real_data[:, :, 6].detach().cpu().numpy().flatten(), 100)

    print('FAKE DATA')
    for i in range(0, fake_data.shape[0], 10000):
        fake_data_deemb = param_emb_container.deembed(fake_data)

        plt.figure('Fake img:' + str(i))
        plt.imshow(fake_data_deemb[i, :, :].split(split_size=img_smpls_to_shw, dim=0)[0].detach().cpu())
        # plt.colorbar()
        plt.ion()
        plt.show()
        plt.figure('Fake hist:' + str(i))
        plt.hist(fake_data_deemb[i, :, 5].detach().cpu().flatten(), 100)
        plt.show()

    plt.pause(0.001)

    real_data.to(device)
    fake_data.to(device)
    param_emb_container.to_device(device)

print('AUTOENCODER')
LATENT_SPACE_SIZE = 1000
autoenc_in, autoenc_out = Autoencoder.create(samples=PADDING, features=EMB_FEATURES, latent=LATENT_SPACE_SIZE,
                                             device=device)
autoenc = Autoencoder(autoenc_in, autoenc_out)
summary(autoenc, input_size=(PADDING, EMB_FEATURES))

autoenc_optimizer = optim.Adam(autoenc.parameters(), lr=0.01)


def train_autoenc():
    for epoch in range(100):

        err: torch.Tensor = torch.Tensor()
        real_data: torch.Tensor = torch.Tensor()
        gen_data: torch.Tensor = torch.Tensor()

        for n_batch, (batch_cat, batch_cont) in enumerate(data_train):
            autoenc_optimizer.zero_grad()
            real_data_cat: torch.Tensor = batch_cat.to(device=device)
            real_data_cont: torch.Tensor = batch_cont.to(device=device)

            real_data = torch.cat([param_emb_container.embed(real_data_cat.long()), real_data_cont], dim=2).detach()

            #show_quality(real_data, gen_data)
            #time.sleep(500.5)

            gen_data, lat_mu, lat_var = autoenc(real_data)

            err = _loss(real_data, gen_data, lat_mu, lat_var)
            err.backward()
            autoenc_optimizer.step()

        print(err.item())
        show_quality(real_data, gen_data)

    torch.save(autoenc_in.state_dict(), parent_path() + 'data/autoencoder_model')


def gen_autoenc_data():
    autoenc_out = AutoencoderOut(samples=PADDING, emb_features=EMB_FEATURES, latent_size=LATENT_SPACE_SIZE, device=device)

    # autoenc_in.load_state_dict(parent_path() + 'data/autoencoder_model')

    np_input = np.random.normal(loc=0, scale=1, size=(100, LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)

    generated_data = autoenc_out.forward(rand_input).detach()
    return generated_data


real_data_out: torch.Tensor = torch.Tensor()
for n_batch, (batch_cat, batch_cont) in enumerate(data_train):
    real_data_cat: torch.Tensor = batch_cat.to(device=device)
    real_data_cont: torch.Tensor = batch_cont.to(device=device)

    real_data = torch.cat([param_emb_container.embed(real_data_cat.long()), real_data_cont], dim=2).detach()

    real_data_out = autoenc(real_data)[0].to(device)
    break


gen_data_out: torch.Tensor = torch.Tensor().cpu()
for i in range(50):
    if i == 0:
        gen_data_out = gen_autoenc_data().cpu()
    else:
        gen_data_out = torch.cat([gen_autoenc_data().cpu(), gen_data_out], dim=0)


show_quality(real_data_out, gen_data_out)

import time
time.sleep(100)
#train_autoenc()
