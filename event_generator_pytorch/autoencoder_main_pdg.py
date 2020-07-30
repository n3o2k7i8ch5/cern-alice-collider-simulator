from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models import ToOneHot
from consts import *
from models_embed_params import ParamEmbedNet

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")



toOneHot = ToOneHot(

        pdg_emb_cnt=PDG_EMB_CNT,
        stat_code_emb_cnt=STAT_CODE_EMB_CNT,
        part_refer_emb_cnt=PART_REFER_EMB_CNT
)

patricles_idx: List[int] = []
for part in particles():
    patricles_idx.append(particles().index(part))

one_hots = toOneHot.pdg_onehot(torch.LongTensor(patricles_idx))

data_train = DataLoader(
    one_hots,
    batch_size=16,
    shuffle=True
)


autoencoder = ParamEmbedNet(
    onehot_size=PDG_EMB_CNT,
    embedding_size=PDG_EMB_DIM,
    device=device
)

#optimizer_emb = optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.1)
optimizer_emb = optim.Adam(autoencoder.parameters(), lr=0.001)

loss = nn.MSELoss()


def train_pdg():
    num_epochs = 2000

    for epoch in range(num_epochs):

        error = None
        for n_batch, batch in enumerate(data_train):
            batch: torch.Tensor = batch.to(device=device)

            result = autoencoder(batch)

            optimizer_emb.zero_grad()

            error = loss(result, batch.float())
            error.backward()

            optimizer_emb.step()

        print(error.item())

        accuracy = 0
        for i in range(len(data_train.dataset)):


            one_hot: torch.Tensor = data_train.dataset[i]
            one_hot = one_hot.unsqueeze(dim=0)
            #print(toOneHot.pdg_reverse(one_hot))
            embedded = autoencoder(one_hot.to(device=device))
            rev = toOneHot.pdg_reverse(embedded)

            if rev.item() == toOneHot.pdg_reverse(one_hot).item():
                accuracy += 1
            #print(rev)

        accuracy /= len(data_train.dataset)
        print()
        print('acc: ' + str(accuracy))
        if accuracy == 1.0:
            torch.save(autoencoder.state_dict(), './autoenc_pdg')
            return


train_pdg()