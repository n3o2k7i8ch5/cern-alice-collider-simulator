import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as func

from common.consts import *
from common.model_pdg_emb_deemb import PDGDeembeder, PDGEmbedder
from common.models_prtc_embed_autoenc import AutoencPrtcl
from common.show_plots import show_plots
from common.show_quality import show_quality
from single_particle_generator_pytorch.load_data import load_data

# device = get_device()

from single_particle_generator_pytorch.train_deembeder import train_deembeder

device = torch.device("cuda")

data = load_data()

data_train = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_deemb_train = DataLoader(
    torch.tensor(particle_idxs()),
    batch_size=len(particle_idxs()),
    shuffle=True
)


def noise(size, device) -> torch.Tensor:
    return torch.randn(size, INP_RAND_SIZE).to(device=device)


def prtcl_loss(input_x, output_x, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> Variable:
    mse_loss = MSELoss()(input_x, output_x)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

    return mse_loss + kld_loss * 0.05


### TRAINING
def show_deemb_quality():
    prtcl_idxs = torch.tensor(particle_idxs(), device=device)

    pdg_onehot = func.one_hot(
        prtcl_idxs,
        num_classes=PDG_EMB_CNT
    ).float()

    emb = embedder(pdg_onehot)
    one_hot_val = deembeder(emb)
    gen_idxs = torch.argmax(one_hot_val, dim=0)  # .item()  # .unsqueeze(dim=2)

    acc = torch.eq(prtcl_idxs, gen_idxs).sum(dim=0).item()

    print('Deembeder acc: ' + str(acc) + '/' + str(len(prtcl_idxs)))


print('AUTOENCODER')

autoenc = AutoencPrtcl(emb_features=EMB_FEATURES, latent_size=PRTCL_LATENT_SPACE_SIZE, device=device)
embedder = PDGEmbedder(PDG_EMB_DIM, PDG_EMB_CNT, device)
deembeder: PDGDeembeder = PDGDeembeder(PDG_EMB_DIM, PDG_EMB_CNT, device)

print(autoenc)
print(deembeder)

autoenc_optimizer = optim.Adam(autoenc.parameters(), lr=0.00002)

EPOCHS = 1


def embed_data(embedder: PDGEmbedder, data):
    cat_data = data[:, :2].long()
    cont_data = data[:, 2:]

    pdg = cat_data[:, 0]
    pdg_onehot = func.one_hot(pdg, num_classes=PDG_EMB_CNT).float()
    prtc_pdg = embedder(pdg_onehot)
    prtc_stat_code = cat_data[:, 1].unsqueeze(dim=1).float()

    emb_cat = torch.cat([prtc_pdg, prtc_stat_code], dim=1)

    return torch.cat([emb_cat, cont_data], dim=1)


def train_autoenc():
    print('TRAINING MODEL:'
          ' BATCH_SIZE = ' + str(BATCH_SIZE) +
          ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
          ', EPOCHS: ' + str(EPOCHS) +
          ', PRTCL_LATENT_SPACE_SIZE: ' + str(PRTCL_LATENT_SPACE_SIZE)
          )

    err: torch.Tensor = torch.Tensor()
    emb_data: torch.Tensor = torch.Tensor()
    gen_data: torch.Tensor = torch.Tensor()

    for epoch in range(EPOCHS):

        for n_batch, batch in enumerate(data_train):
            autoenc_optimizer.zero_grad()

            real_data: torch.Tensor = batch.to(device=device).detach()
            emb_data = embed_data(embedder, real_data)

            gen_data, lat_mean, lat_logvar, lat_vec = autoenc(emb_data)

            err = prtcl_loss(
                input_x=emb_data,
                output_x=gen_data,
                lat_mean=lat_mean,
                lat_logvar=lat_logvar)

            err.backward()
            autoenc_optimizer.step()

            if n_batch % 4_000 == 0:
                show_deemb_quality()
                print('Batch: ' + str(n_batch) + '/' + str(len(data_train)))
                show_quality(emb_data, gen_data, feature_range=(-10, -5))
                print('Error: ' + str(err.item()))

        print(err.item())
        show_quality(emb_data, gen_data, feature_range=(-10, -5))
        show_deemb_quality()

    print('Training deembeder')
    train_deembeder(
        deembeder=deembeder,
        embedder=embedder,
        epochs=100,
        device=device
    )

    torch.save(deembeder.state_dict(), parent_path() + 'data/pdg_deembed_model')
    torch.save(autoenc.state_dict(), parent_path() + 'data/single_prtc_autoenc_model')
    show_real_gen_data_comparison()


def gen_autoenc_data(autoenc):
    np_input = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE, PRTCL_LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    generated_data = autoenc.deembed(rand_input).detach()
    return generated_data


def show_real_gen_data_comparison():
    SIZE = 100

    print('GENERATING DATA'
          ' BATCH_SIZE = ' + str(BATCH_SIZE))

    autoenc.load_state_dict(torch.load(parent_path() + 'data/single_prtc_autoenc_model'))

    for param in autoenc.parameters():
        param.requires_grad = False

    all_emb_data: torch.Tensor = torch.Tensor()
    gen_data: torch.Tensor = torch.Tensor()

    for n_batch, batch in enumerate(data_train):
        real_data = batch.to(device)

        emb_data = embed_data(embedder, real_data)

        if n_batch == 0:
            all_emb_data = emb_data.cpu()
        else:
            all_emb_data = torch.cat([emb_data.cpu(), all_emb_data], dim=0)

        if n_batch == SIZE:
            break

    for i in range(SIZE):
        if i == 0:
            gen_data = gen_autoenc_data(autoenc).cpu()
        else:
            gen_data = torch.cat([gen_autoenc_data(autoenc).cpu(), gen_data], dim=0)

    for param in autoenc.parameters():
        param.requires_grad = True

    show_quality(all_emb_data, gen_data, feature_range=(-10, -5))

train_autoenc()
# show_real_gen_data_comparison()
