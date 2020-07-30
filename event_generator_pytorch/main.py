import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as func

import matplotlib.pyplot as plt

from common.consts import *
from event_generator_pytorch.load_data import load_data
from event_generator_pytorch.models_embedding_autoencoder import Autoencoder, PDGDeembeder

PADDING = 20

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


def _loss(input_x, output_x, lat_mu, lat_var, real_one_hot, gen_one_hot) -> Variable:
    mse_loss = MSELoss()(input_x, output_x)

    # kl_loss = 0.5 * torch.sum(lat_var.exp() + lat_mu.pow(2) - 1.0 - lat_var)

    kl_loss = MSELoss()(
        lat_mu,
        torch.zeros(size=lat_mu.size()).to(device)
    ) + MSELoss()(
        lat_var,
        torch.ones(size=lat_var.size()).to(device)
    )

    deemb_loss = MSELoss()(real_one_hot, gen_one_hot)

    # print('mse_loss ' + str(mse_loss.item()))
    # print('kl_loss ' + str(kl_loss.item()))

    #return deemb_loss
    return mse_loss + (kl_loss / 10) + deemb_loss


### TRAINING


def show_quality(real_data, fake_data):

    IMG_SAMPLES_TO_SHOW = 80

    '''
    plt.figure()
    plt.title('Real img')
    plt.imshow(real_data[0, :, :].split(split_size=IMG_SAMPLES_TO_SHOW, dim=0)[0])
    plt.colorbar()
    plt.ion()
    '''

    '''
    plt.figure()
    plt.title('Fake img')
    plt.imshow(fake_data[0, :, :].split(split_size=IMG_SAMPLES_TO_SHOW, dim=0)[0])
    plt.colorbar()
    plt.ion()
    '''

    for h_lay_idx in range(CONT_FEATURES + CAT_FEATURES):

        real_data = real_data.detach().cpu()
        fake_data = fake_data.detach().cpu()

        plt.figure()
        plt.title('Real hist')
        plt.hist(real_data[:, :, h_lay_idx].flatten(), 50, alpha=0.5, color='b', label='real')
        plt.hist(fake_data[:, :, h_lay_idx].flatten(), 50, alpha=0.5, color='r', label='gen')
        plt.legend(prop={'size': 10})
        plt.show()

        plt.pause(0.001)

    real_data.to(device)
    fake_data.to(device)


def show_deemb_quality():
    prtcl_idxs = particle_idxs()

    acc = 0
    for pdg_code in prtcl_idxs:
        pdg_onehot = func.one_hot(
            torch.tensor(pdg_code).to(device),
            num_classes=PDG_EMB_CNT
        ).float()
        emb = autoenc.auto_in.pdg_emb(pdg_onehot)
        one_hot_val = deembeder(emb)
        gen_val = torch.argmax(one_hot_val, dim=0).cpu().item()  # .unsqueeze(dim=2)

        if pdg_code == gen_val:
            acc += 1

    print('Deembeder acc: ' + str(acc) + '/' + str(len(prtcl_idxs)))

print('AUTOENCODER')
LATENT_SPACE_SIZE = 82 #48
autoenc_in, autoenc_out = Autoencoder.create(samples=PADDING, emb_features=EMB_FEATURES, latent=LATENT_SPACE_SIZE,
                                             device=device)
autoenc = Autoencoder(autoenc_in, autoenc_out)
# summary(autoenc, input_size=(PADDING, FEATURES))

deembeder: PDGDeembeder = PDGDeembeder(PDG_EMB_DIM, PDG_EMB_CNT, device)

# Z jakiegoś arcyciekawego powodu Adam działa świetnie, a SGD prawie w ogóle.
# Chodzi głównie o Deembeder, który na SGD się nie chce uczyć.
autoenc_optimizer = optim.Adam(params=autoenc.parameters(), lr=0.0002)
deembed_optimizer = optim.Adam(params=deembeder.parameters(), lr=0.0002)


def train_autoenc():
    for epoch in range(200):

        err: torch.Tensor = torch.Tensor()
        emb_data: torch.Tensor = torch.Tensor()
        gen_data: torch.Tensor = torch.Tensor()
        lat_mu: torch.Tensor = torch.Tensor()
        lat_var: torch.Tensor = torch.Tensor()

        for n_batch, (batch_cat, batch_cont) in enumerate(data_train):
            autoenc_optimizer.zero_grad()
            deembed_optimizer.zero_grad()

            # TRAIN V-AUTOENCODER
            real_data_cat: torch.Tensor = batch_cat.to(device=device)
            real_data_cont: torch.Tensor = batch_cont.to(device=device)

            real_data = torch.cat([real_data_cat, real_data_cont], dim=2).detach()

            # show_quality(real_data, gen_data)
            # time.sleep(500.5)

            emb_data, gen_data, lat_mu, lat_var = autoenc(real_data)

            # TRAIN DEEMBEDER
            real_one_hot = None
            gen_one_hot = None
            for pgd_idxs in data_deemb_train: # should fire only one time

                real_one_hot = func.one_hot(pgd_idxs, num_classes=PDG_EMB_CNT).float()
                embed = autoenc.auto_in.pdg_emb(real_one_hot)
                gen_one_hot = deembeder(embed)

            err = _loss(emb_data, gen_data, lat_mu, lat_var, real_one_hot, gen_one_hot)
            err.backward()
            autoenc_optimizer.step()
            deembed_optimizer.step()

        show_deemb_quality()

        print(err.item())

    torch.save(autoenc.state_dict(), parent_path() + 'data/event_autoenc_model')
    torch.save(deembeder.state_dict(), parent_path() + 'data/event_pdg_deembed_model')


def generate_data(autoenc_out, deembeder: PDGDeembeder):
    np_input = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE, LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    gen_emb_data = autoenc_out.forward(rand_input).detach()

    gen_data = deembeder.deemb(gen_emb_data)

    return gen_data


def show_real_gen_data_comparison():
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

    show_quality(real_data, gen_data_out)

    import time
    time.sleep(1000)

# train_autoenc()
show_real_gen_data_comparison()
