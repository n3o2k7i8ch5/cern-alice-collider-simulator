import torch
from torch import optim
from torch.nn import MSELoss
from torch.nn.functional import one_hot


from common.consts import PDG_EMB_CNT, particle_idxs
from common.model_pdg_emb_deemb import PDGDeembeder


def train_deembeder(deembeder: PDGDeembeder, epochs: int, embedder, device):

    deembed_optimizer = optim.Adam(deembeder.parameters(), lr=0.0001)
    deemb_loss = MSELoss()

    real_one_hot = one_hot(torch.tensor(particle_idxs(), device=device), num_classes=PDG_EMB_CNT).float()

    for i in range(epochs):
        deembed_optimizer.zero_grad()
        embed = embedder(real_one_hot)
        gen_one_hot = deembeder(embed)
        err_deemb = deemb_loss(real_one_hot, gen_one_hot)
        err_deemb.backward()
        deembed_optimizer.step()
        print(err_deemb.item())