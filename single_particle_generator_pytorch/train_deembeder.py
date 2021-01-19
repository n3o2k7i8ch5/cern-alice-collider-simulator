import torch
from torch import optim, nn
from torch.nn import MSELoss, BCELoss
from torch.nn.functional import one_hot


from common.consts import PDG_EMB_CNT, particle_idxs
from common.model_pdg_emb_deemb import PDGDeembedder, PDGEmbedder


def train_deembeder(deembedder: PDGDeembedder, epochs: int, embedder: PDGEmbedder, device):

    deembed_optimizer = optim.Adam(deembedder.parameters(), lr=1e-5)
    deemb_loss = nn.MSELoss()

    real_one_hot = one_hot(torch.tensor(particle_idxs(), device=device), num_classes=PDG_EMB_CNT).float()

    for param in embedder.parameters():
        param.requires_grad = False

    for i in range(epochs):
        deembed_optimizer.zero_grad()
        embed = embedder(real_one_hot)
        gen_one_hot = deembedder(embed)
        err_deemb = deemb_loss(real_one_hot, gen_one_hot)
        err_deemb.backward()
        deembed_optimizer.step()

        gen_one_hot = gen_one_hot > .5
        gen_one_hot = gen_one_hot.int()

        acc = 0
        if i%100 == 99:
            # calc accuracy
            diffs = torch.eq(real_one_hot, gen_one_hot).all(dim=1).int()
            size = len(diffs)
            acc += diffs.int().sum()
            acc /= size

            print(f'acc: {acc}, err: {err_deemb.item()}')

        for param in embedder.parameters():
            param.requires_grad = True