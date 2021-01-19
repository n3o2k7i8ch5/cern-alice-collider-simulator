import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from common.consts import *
from common.load_event_data import load_data
from common.model_pdg_emb_deemb import PDGDeembedder, show_deemb_quality
from common.models_prtc_embed_autoenc import AutoencPrtcl

import torch.nn.functional as func

from common.show_plots import show_plots
from prtcl_based_event_gen_pytorch.models import AutoencPrtclBasedEvnt

EVNT_LTNT_SPACE_SIZE = 200

AE_MAIN_MODEL_PATH = parent_path() + 'data/prtcl_based_event_gen_ae_main_model'
AE_PRTCL_MODEL_PATH = parent_path() + 'data/prtcl_based_event_gen_ae_prtcl_model'
PDG_DEMBED_MODEL_PATH = parent_path() + 'data/prtcl_based_event_gen_pdg_deembed_model'

data_cat, data_cont, max_length = load_data()

padded_data_cat: torch.Tensor = pad_sequence(data_cat, batch_first=True).split(split_size=PADDING, dim=1)[0]
padded_data_cont: torch.Tensor = pad_sequence(data_cont, batch_first=True).split(split_size=PADDING, dim=1)[0]

data_train = DataLoader(
    TensorDataset(padded_data_cat.float(), padded_data_cont),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print('AUTOENCODER')

device = torch.device("cuda")

''' # VAE
def ae_prtcl_loss(input_x, output_x, lat_mu, lat_var, show_partial=False) -> Variable:
'''
def ae_prtcl_loss(input_x, output_x, show_partial=False) -> Variable:
    mse_loss = MSELoss()(input_x, output_x)
    ''' # VAE
    kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_var - lat_mu ** 2 - lat_var.exp(), dim=1), dim=0)
    '''

    if show_partial:
        print('\tPRTCL mse_loss ' + str(mse_loss.item()))
        # print('\tPRTCL kld_loss ' + str(kld_loss.item()))

    return mse_loss # + kld_loss * 0.001


def ae_main_loss(input, output, lat_mu, lat_var, show_partial=False) -> Variable:
    mse_loss = MSELoss()(input, output)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_var - lat_mu ** 2 - lat_var.exp(), dim=1), dim=0)

    if show_partial:
        print('\tMAIN mse_loss ' + str(mse_loss.item()))
        print('\tMAIN kld_loss ' + str(kld_loss.item()))

    return mse_loss + kld_loss*0.00


def dembed_loss(real_one_hot, gen_one_hot, show_partial=False):
    deemb_loss = MSELoss()(real_one_hot, gen_one_hot)

    if show_partial:
        print('\tDEEMB deemb_loss ' + str(deemb_loss.item()))

    return deemb_loss


ae_prtcl_in, ae_prtcl_out = AutoencPrtcl.create(emb_features=EMB_FEATURES, latent=PRTCL_LATENT_SPACE_SIZE, device=device)
ae_prtcl = AutoencPrtcl(ae_prtcl_in, ae_prtcl_out)

ae_prtcl.load_state_dict(torch.load(parent_path() + 'data/single_prtc_autoenc_model'))

ae_main_in, ae_main_out = AutoencPrtclBasedEvnt.create(
    emb_size=EMB_FEATURES,
    prtcl_ltnt_size=PRTCL_LATENT_SPACE_SIZE,
    prtcl_in_evnt_cnt=PADDING,
    evnt_ltnt_size=EVNT_LTNT_SPACE_SIZE,
    device=device)
ae_main = AutoencPrtclBasedEvnt(ae_main_in, ae_main_out)

deembeder: PDGDeembedder = PDGDeembedder(PDG_EMB_DIM, PDG_EMB_CNT, device)

print('AE_MAIN_IN')
summary(ae_main_in, input_size=(PRTCL_LATENT_SPACE_SIZE * PADDING,))
print('AE_MAIN_OUT')
summary(ae_main_out, input_size=(EVNT_LTNT_SPACE_SIZE,))
print('AE_PRTCL_IN')
summary(ae_prtcl_in, input_size=(FEATURES,))
print('AE_PRTCL_OUT')
summary(ae_prtcl_out, input_size=(PRTCL_LATENT_SPACE_SIZE,))
print('DEEMBEDER')
summary(deembeder, input_size=(PDG_EMB_DIM,))

# Z jakiegoś arcyciekawego powodu Adam działa świetnie, a SGD prawie w ogóle.
# Chodzi głównie o Deembeder, który na SGD się nie chce uczyć.
ae_prtcl_optimizer = optim.Adam(ae_prtcl.parameters(), lr=0.001)
ae_main_optimizer = optim.Adam(ae_main.parameters(), lr=0.001)
deembed_optimizer = optim.Adam(deembeder.parameters(), lr=0.0001)

EPOCHS = 50 #200


def train_autoenc():
    print('TRAINING MODEL:'
          ' BATCH_SIZE = ' + str(BATCH_SIZE) +
          ', PADDING = ' + str(PADDING) +
          ', PARTICLE_DIM: ' + str(PARTICLE_DIM) +
          ', EPOCHS: ' + str(EPOCHS) +
          ', EVNT_LTNT_SPACE_SIZE: ' + str(EVNT_LTNT_SPACE_SIZE)
          )

    for epoch in range(EPOCHS):

        '''
        emb_data: torch.Tensor = torch.Tensor()
        gen_data: torch.Tensor = torch.Tensor()
        lat_mu: torch.Tensor = torch.Tensor()
        lat_var: torch.Tensor = torch.Tensor()
        gen_one_hot: torch.Tensor = torch.Tensor()
        '''

        err: torch.Tensor = torch.Tensor()
        all_pdg_real_one_hot = func.one_hot(
            torch.tensor(particle_idxs(), device=device),
            num_classes=PDG_EMB_CNT
        ).float()

        for n_batch, (batch_cat, batch_cont) in enumerate(data_train):
            ae_prtcl_optimizer.zero_grad()
            deembed_optimizer.zero_grad()
            ae_main_optimizer.zero_grad()

            # TRAIN V-AUTOENCODER
            real_data_cat: torch.Tensor = batch_cat.to(device=device)
            real_data_cont: torch.Tensor = batch_cont.to(device=device)

            real_data = torch.cat([real_data_cat, real_data_cont], dim=2).detach()

            # # #
            # Particle params to particle latent space
            #
            batch_size = len(real_data)
            inp_prtcl = real_data.flatten(start_dim=0, end_dim=1)

            ''' # VAE
            inp_prtcl_emb, out_prtcl_emb, lat_mu_prtcl, lat_var_prtcl, in_prtcl_ltnts = ae_prtcl(inp_prtcl)
            '''

            inp_prtcl_emb, out_prtcl_emb, in_prtcl_ltnts = ae_prtcl(inp_prtcl)

            in_prtcl_ltnts: torch.Tensor = in_prtcl_ltnts.reshape(batch_size, PADDING, PRTCL_LATENT_SPACE_SIZE)
            #
            #
            # # #

            out_prtcl_ltnts, lat_main_mu, lat_main_var = ae_main(in_prtcl_ltnts.detach())

            # TRAIN DEEMBEDER
            #pdg_embs = ae_prtcl_in.pdg_emb(all_pdg_real_one_hot)
            #gen_one_hot = deembeder(pdg_embs)

            # Errors and backprop
            ''' # VAE
            err_prtcl = ae_prtcl_loss(inp_prtcl_emb, out_prtcl_emb, lat_mu_prtcl, lat_var_prtcl, show_partial=n_batch == 0)
            '''

            err_prtcl = ae_prtcl_loss(inp_prtcl_emb, out_prtcl_emb, show_partial=n_batch == 0)

            err_main = ae_main_loss(
                input=in_prtcl_ltnts,
                output=out_prtcl_ltnts,
                lat_mu=lat_main_mu,
                lat_var=lat_main_var,
                show_partial=n_batch == 0)
            #err_deemb = dembed_loss(all_pdg_real_one_hot, gen_one_hot, show_partial=n_batch == 0)

            '''
            if epoch < 7:
                err = err_prtcl + err_main + err_deemb
            else:
                err = err_main + err_deemb
            '''
            err = err_prtcl + err_main # + err_deemb
            err.backward()

            #if epoch < 7:
            ae_prtcl_optimizer.step()

            ae_main_optimizer.step()

            #deembed_optimizer.step()

        show_deemb_quality(ae_prtcl_in.pdg_emb, deembeder, device)

        print('epoch: ' + str(epoch) + ', ' + str(err.item()))

    print('SAVING MODELS...')
    torch.save(ae_main.state_dict(), AE_MAIN_MODEL_PATH)
    torch.save(ae_prtcl.state_dict(), AE_PRTCL_MODEL_PATH)
    torch.save(deembeder.state_dict(), PDG_DEMBED_MODEL_PATH)

    show_real_gen_data_comparison()


def generate_data(ae_main_out, ae_prtcl_out, deembeder: PDGDeembedder):
    np_input = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE, EVNT_LTNT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    gen_ltnt_data = ae_main_out.forward(rand_input).detach()

    batch_size = len(gen_ltnt_data)
    gen_ltnt_data = gen_ltnt_data.reshape(batch_size * PADDING, PRTCL_LATENT_SPACE_SIZE)
    gen_emb_data = ae_prtcl_out.forward(gen_ltnt_data)

    gen_emb_data = gen_emb_data.reshape(batch_size, PADDING, EMB_FEATURES)

    gen_data = deembeder.deemb(gen_emb_data)

    return gen_data


def show_real_gen_data_comparison(load: bool = True):
    print('GENERATING DATA')

    if load:
        ae_main.load_state_dict(torch.load(AE_MAIN_MODEL_PATH))
        ae_prtcl.load_state_dict(torch.load(AE_PRTCL_MODEL_PATH))
        deembeder.load_state_dict(torch.load(PDG_DEMBED_MODEL_PATH))

    for param in ae_main.parameters():
        param.requires_grad = False

    for param in ae_prtcl.parameters():
        param.requires_grad = False

    for param in deembeder.parameters():
        param.requires_grad = False

    real_data: torch.Tensor = torch.Tensor()

    BATCHES_TO_DISP = 25

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
            gen_data_out = generate_data(ae_main.auto_out, ae_prtcl.auto_out, deembeder).cpu()
        else:
            gen_data_out = torch.cat(
                [generate_data(ae_main.auto_out, ae_prtcl.auto_out, deembeder).cpu(), gen_data_out], dim=0)

    show_plots(real_data, gen_data_out)

    for param in ae_main.parameters():
        param.requires_grad = True

    for param in ae_prtcl.parameters():
        param.requires_grad = True

    for param in deembeder.parameters():
        param.requires_grad = True


train_autoenc()
# show_real_gen_data_comparison()
