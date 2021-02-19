from typing import Tuple

import torch
from torch import nn

from berka_vae.consts import CRED_EMBED_DIM, DEBT_EMBED_DIM
from common.sample_norm import sample_norm

'''
Takes in a batch of particles, each of size:
1 x FEATURES

Returns:
a) a batch of embedded input particles, each of size:
   1 x EMB_FEATURES

b) a batch of embedded output particles, each of size;
   1 x EMB_FEATURES

c) a batch of var and mu (for VAE KLD calculation), each of size:
   1 x PRTCL_LATENT_SPACE_SIZE
'''


class VAE(nn.Module):

    def one_hot_generator(self, in_features, out_features, device):
        return nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_features),
            nn.Softmax(),
        ).to(device=device)

    def __init__(
            self,
            emb_features,
            latent_size,

            cat_trans_types_cnt: int,
            cat_trans_ops_cnt: int,
            cat_trans_descs_cnt: int,
            cat_bank_code_debtor_cnt: int,

            device
    ):
        super(VAE, self).__init__()

        self.emb_features = emb_features
        self.latent_size = latent_size

        self.mean = nn.Sequential(
            nn.Linear(emb_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, latent_size),
        ).to(device=device)

        self.logvar = nn.Sequential(
            nn.Linear(emb_features, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, latent_size),
        ).to(device=device)

        self.decoder_trans_types = self.one_hot_generator(latent_size, cat_trans_types_cnt, device)
        self.decoder_trans_ops = self.one_hot_generator(latent_size, cat_trans_ops_cnt, device)
        self.decoder_trans_descs = self.one_hot_generator(latent_size, cat_trans_descs_cnt, device)
        self.decoder_bank_code_debtor = self.one_hot_generator(latent_size, cat_bank_code_debtor_cnt, device)

        self.decoder_log_balance = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ).to(device=device)

        self.decoder_perc_withdrawed = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device=device)

        remain_features = self.emb_features - (
                    2 + cat_trans_types_cnt + cat_trans_ops_cnt + cat_trans_descs_cnt + cat_bank_code_debtor_cnt)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, remain_features),
            nn.Sigmoid(),
        ).to(device=device)

    def encode(self, x: torch.Tensor):
        lat_mean = self.mean(x)
        lat_logvar = self.logvar(x)

        return lat_mean, lat_logvar

    def decode_parts(self, lat_vec: torch.Tensor) -> Tuple:

        emb_ids = self.decoder(lat_vec)
        cred_emb_id = emb_ids[:, :CRED_EMBED_DIM]
        debt_emb_id = emb_ids[:, :DEBT_EMBED_DIM]

        return cred_emb_id, \
               debt_emb_id, \
               \
               self.decoder_log_balance(lat_vec), \
               self.decoder_perc_withdrawed(lat_vec), \
               \
               self.decoder_trans_types(lat_vec), \
               self.decoder_trans_ops(lat_vec), \
               self.decoder_trans_descs(lat_vec), \
               self.decoder_bank_code_debtor(lat_vec)

    def decode(self, lat_vec: torch.Tensor):
        return torch.cat(self.decode_parts(lat_vec), dim=1)

    def forward(self, x: torch.Tensor):
        lat_mean, lat_logvar = self.encode(x)
        lat_vec = sample_norm(mean=lat_mean, logvar=lat_logvar)

        out_emb = self.decode_parts(lat_vec)
        return out_emb, lat_mean, lat_logvar, lat_vec
