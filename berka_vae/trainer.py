import math
from typing import List

import torch
from torch.autograd import Variable
from torch.nn import MSELoss, BCELoss
import numpy as np

from berka_vae.consts import CRED_EMBED_DIM, DEBT_EMBED_DIM
from berka_vae.models.vae import VAE
from common.consts import parent_path
from i_trainer.i_trainer import ITrainer

import pandas as pd

from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder
from single_prtcl_generator_vae_pytorch.models.pdg_embedder import PDGEmbedder


class Trainer(ITrainer):
    BATCH_SIZE = 256
    LATENT_SIZE = 24

    NAN_VAL_TRAIN_STR = '__NAN_VAL__'

    show_feat_rng = -7, -4

    MODEL_SAVE_PATH = parent_path() + 'data/berka_vae.model'
    EMB_CRED_SAVE_PATH = parent_path() + 'data/embed_cred.model'
    EMB_DEBT_SAVE_PATH = parent_path() + 'data/embed_debt.model'
    DEEMB_CRED_SAVE_PATH = parent_path() + 'data/deembed_cred.model'
    DEEMB_DEBT_SAVE_PATH = parent_path() + 'data/deembed_debt.model'

    def cut_emb_data(self, emb_data):
        start_idx = 0

        cred_id = emb_data[:, start_idx:start_idx + CRED_EMBED_DIM]
        start_idx += CRED_EMBED_DIM

        debt_id = emb_data[:, start_idx:start_idx + DEBT_EMBED_DIM]
        start_idx += DEBT_EMBED_DIM

        df_acc_log_balance = emb_data[:, start_idx:start_idx + 1]
        start_idx += 1

        df_perc_withdrawed = emb_data[:, start_idx:start_idx + 1]
        start_idx += 1

        df_trans_types = emb_data[:, start_idx:start_idx + len(self.cols_trans_types)]
        start_idx += len(self.cols_trans_types)

        df_trans_ops = emb_data[:, start_idx:start_idx + len(self.cols_trans_ops)]
        start_idx += len(self.cols_trans_ops)

        df_trans_descs = emb_data[:, start_idx:start_idx + len(self.cols_trans_descs)]
        start_idx += len(self.cols_trans_descs)

        df_bank_code_debtor = emb_data[:, start_idx:start_idx + len(self.cols_bank_code_debtor)]
        start_idx += len(self.cols_bank_code_debtor)

        return cred_id, \
               debt_id, \
               df_acc_log_balance, \
               df_perc_withdrawed, \
               df_trans_types, \
               df_trans_ops, \
               df_trans_descs, \
               df_bank_code_debtor

    def load_trans_data(self, device=None):

        data = pd.read_csv('berka_trans_mapped.csv', sep='\t')
        data.drop(data.columns[0], axis=1, inplace=True)
        data.drop('trans_id', axis=1, inplace=True)

        self.uniq_cred = data['acc_id_creditor'].unique().tolist()
        self.uniq_cred_idxs = [x for x in range(len(self.uniq_cred))]
        self.uniq_debt = data['acc_id_debtor'].unique().tolist()
        self.uniq_debt_idxs = [x for x in range(len(self.uniq_debt))]

        cred_idxs = data['acc_id_creditor'].apply(func=lambda item: self.uniq_cred.index(item))
        debt_idxs = data['acc_id_debtor'].apply(func=lambda item: 0 if math.isnan(item) else self.uniq_debt.index(item))

        df_trans_types = pd.get_dummies(data['trans_type'], prefix_sep='$')
        self.cols_trans_types = df_trans_types.columns

        df_trans_ops = pd.get_dummies(data['trans_operation'], prefix_sep='$')
        self.cols_trans_ops = df_trans_ops.columns

        df_trans_amount = data['trans_amount']
        df_acc_balance = data['acc_balance']

        df_acc_log_balance = df_acc_balance.abs().apply(func=lambda x: math.log(1 + x, 5))

        df_perc_withdrawed = df_trans_amount / df_acc_balance

        _trans_desc = data['trans_description'].fillna(Trainer.NAN_VAL_TRAIN_STR)
        df_trans_descs = pd.get_dummies(_trans_desc, prefix_sep='$')
        self.cols_trans_descs = df_trans_descs.columns

        _bank_code_debtor = data['bank_code_debtor'].fillna(Trainer.NAN_VAL_TRAIN_STR)
        df_bank_code_debtor = pd.get_dummies(_bank_code_debtor, prefix_sep='$')
        self.cols_bank_code_debtor = df_bank_code_debtor.columns

        result = pd.concat([
            cred_idxs,
            debt_idxs,

            df_acc_log_balance,
            df_perc_withdrawed,

            df_trans_types,
            df_trans_ops,
            df_trans_descs,
            df_bank_code_debtor,
        ], axis=1, join="inner")

        result = result.to_numpy()

        if device is None:
            device = self.device

        return torch.tensor(result, dtype=torch.float32, device=device)

    def loss(self,
             input_x,
             output_x,

             lat_mean: torch.Tensor,
             lat_logvar: torch.Tensor
             ) -> (Variable, Variable, Variable):

        real_cred_id, \
        real_debt_id, \
        real_log_balance, \
        real_perc_withdrawed, \
        real_trans_types, \
        real_trans_ops, \
        real_trans_descs, \
        real_bank_code_debtor = self.cut_emb_data(input_x)

        gen_cred_id, \
        gen_debt_id, \
        gen_log_balance, \
        gen_perc_withdrawed, \
        gen_trans_types, \
        gen_trans_ops, \
        gen_trans_descs, \
        gen_bank_code_debtor = output_x

        loss_cred_id = MSELoss()(real_cred_id, gen_cred_id)
        loss_debt_id = MSELoss()(real_debt_id, gen_debt_id)
        loss_log_balance = MSELoss()(real_log_balance, gen_log_balance)
        loss_perc_withdrawed = MSELoss()(real_perc_withdrawed, gen_perc_withdrawed)
        loss_trans_types = BCELoss()(gen_trans_types, real_trans_types.detach())
        loss_trans_ops = BCELoss()(gen_trans_ops, real_trans_ops.detach())
        loss_trans_descs = BCELoss()(gen_trans_descs, real_trans_descs.detach())
        loss_bank_code_debtor = BCELoss()(gen_bank_code_debtor, real_bank_code_debtor.detach())

        recon_loss = loss_cred_id + \
                     loss_debt_id + \
                     5*loss_log_balance + \
                     loss_perc_withdrawed + \
                     loss_trans_types + \
                     loss_trans_ops + \
                     loss_trans_descs + \
                     loss_bank_code_debtor

        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return recon_loss + kld_loss * 1e-4, recon_loss, kld_loss

    def create_vae(self, load) -> VAE:
        vae = VAE(
            emb_features=74,
            latent_size=Trainer.LATENT_SIZE,

            cat_trans_types_cnt=len(self.cols_trans_types),
            cat_trans_ops_cnt=len(self.cols_trans_ops),
            cat_trans_descs_cnt=len(self.cols_trans_descs),
            cat_bank_code_debtor_cnt=len(self.cols_bank_code_debtor),

            device=self.device
        )
        if load:
            vae.load_state_dict(torch.load(Trainer.MODEL_SAVE_PATH))

        return vae

    def embed_data(self, data, embedders: List[PDGEmbedder]):
        cat_data = data[:, :2].long()
        cont_data = data[:, 2:].float()

        embedder_cred = embedders[0]
        embedder_debt = embedders[1]

        cred_idxs = cat_data[:, 0]
        emb_cred_idxs = embedder_cred(cred_idxs)

        debt_idxs = cat_data[:, 1]
        emb_debt_idxs = embedder_debt(debt_idxs)

        return torch.cat([emb_cred_idxs, emb_debt_idxs, cont_data], dim=1)

    def train(self, epochs, load=False):

        _data = self.load_trans_data()
        data_train, data_valid = self.prep_data(_data, batch_size=Trainer.BATCH_SIZE, valid=0.1)

        embedder_cred = PDGEmbedder(num_embeddings=len(self.uniq_cred), embedding_dim=CRED_EMBED_DIM,
                                    device=self.device)
        embedder_debt = PDGEmbedder(num_embeddings=len(self.uniq_debt), embedding_dim=DEBT_EMBED_DIM,
                                    device=self.device)

        deembedder_cred = PDGDeembedder(CRED_EMBED_DIM, len(self.uniq_cred), self.device)
        deembedder_debt = PDGDeembedder(DEBT_EMBED_DIM, len(self.uniq_debt), self.device)

        vae = self.create_vae(load=load)
        autoenc_optimizer = torch.optim.Adam(vae.parameters(), lr=0.00005)

        uniq_cred_idxs = torch.tensor(self.uniq_cred_idxs, device=self.device)
        uniq_debt_idxs = torch.tensor(self.uniq_debt_idxs, device=self.device)

        if load:
            print('LOADING MODEL STATES...')

            # autoenc.load_state_dict(torch.load(Trainer.AUTOENC_SAVE_PATH))
            # embedder.load_state_dict(torch.load(Trainer.PDG_EMBED_SAVE_PATH))
            # deembedder.load_state_dict(torch.load(Trainer.PDG_DEEMBED_SAVE_PATH))

        print('VAE')
        print(vae)
        print('EMBEDDER CRED')
        print(embedder_cred)
        print('EMBEDDER DEBT')
        print(embedder_debt)
        print('DEEMBEDDER CRED')
        print(deembedder_cred)
        print('DEEMBEDDER DEBT')
        print(deembedder_debt)

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):
                autoenc_optimizer.zero_grad()

                real_data: torch.Tensor = batch.to(self.device).detach()

                emb_data = self.embed_data(real_data, [embedder_cred, embedder_debt])

                gen_data, lat_mean, lat_logvar, lat_vec = vae(emb_data)

                loss, mse_loss, kld_loss = self.loss(
                    input_x=emb_data,
                    output_x=gen_data,
                    lat_mean=lat_mean,
                    lat_logvar=lat_logvar)

                loss.backward()
                autoenc_optimizer.step()

                if n_batch % 50 == 0:
                    deemb_accs = self.train_deembeders([
                        (uniq_cred_idxs, embedder_cred, deembedder_cred),
                        (uniq_debt_idxs, embedder_debt, deembedder_debt)],
                        epochs=2)

                if n_batch % 500 == 0:
                    # show_lat_histograms(lat_mean=lat_mean, lat_logvar=lat_logvar)
                    self.show_deemb_quality(uniq_cred_idxs, embedder_cred, deembedder_cred)
                    self.show_deemb_quality(uniq_debt_idxs, embedder_debt, deembedder_debt)

                    valid_loss = self._valid_loss(vae, embedder_cred, embedder_debt, data_valid)



                    # show_quality(emb_data, gen_data, feature_range=self.show_feat_rng, save=True)
                    self.show_img_comparison(emb_data, torch.cat(gen_data, dim=1), title='Training')

                    self.show_real_gen_data_comparison(
                        vae,
                        real_data,
                        [embedder_cred, embedder_debt],
                        emb=True,
                        show_histograms=False,
                        sample_cnt=30,
                        deembedder=deembedder_cred,
                        save=True
                    )

                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'train loss: {"{:.6f}".format(round(loss.item(), 6))} :: '
                        f'kld loss: {"{:.6f}".format(round(kld_loss.item(), 6))} :: '
                        f'mse loss: {"{:.6f}".format(round(mse_loss.item(), 6))} :: '
                        f'valid loss: {"{:.6f}".format(round(valid_loss, 6))} :: '
                        f'deemb accs: {[acc.item() for acc in deemb_accs]}'
                    )

                    self._save_models(vae, embedder_cred, embedder_debt, deembedder_cred, deembedder_debt)

        return vae, embedder_cred, embedder_debt, deembedder_cred, deembedder_debt

    def _save_models(self, autoenc, emb_cred, emb_debt, deemb_cred, deemb_debt):
        print('Saving autoencoder model')
        torch.save(autoenc.state_dict(), Trainer.MODEL_SAVE_PATH)

        print('Saving embed cred model')
        torch.save(emb_cred.state_dict(), Trainer.EMB_CRED_SAVE_PATH)

        print('Saving embed debt model')
        torch.save(emb_debt.state_dict(), Trainer.EMB_DEBT_SAVE_PATH)

        print('Saving deembed cred model')
        torch.save(deemb_cred.state_dict(), Trainer.DEEMB_CRED_SAVE_PATH)

        print('Saving deembed debt model')
        torch.save(deemb_debt.state_dict(), Trainer.DEEMB_DEBT_SAVE_PATH)

    def _valid_loss(self, model, embedder_cred: PDGEmbedder, embedder_debt, valid_data_loader) -> (float, float):
        loss = 0

        criterion = MSELoss()

        for batch_data in valid_data_loader:
            emb_data = self.embed_data(batch_data.to(self.device), [embedder_cred, embedder_debt])
            out_emb, lat_mean, lat_vec, lat_vec = model(emb_data)
            out_emb = torch.cat(out_emb, dim=1)
            train_loss = criterion(out_emb, emb_data)
            loss += train_loss.item()
        loss /= len(valid_data_loader)

        return loss

    def gen_autoenc_data(self, sample_cnt, model: VAE):

        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, Trainer.LATENT_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = model.decode_parts(rand_input)
        return generated_data

    def convert_to_dataframe(self, gen_data, deemb_cred, deemb_debt):
        gen_cred_id, \
        gen_debt_id, \
        gen_log_balance, \
        gen_perc_withdrawed, \
        gen_trans_types, \
        gen_trans_ops, \
        gen_trans_descs, \
        gen_bank_code_debtor = gen_data

        cred_idxs = deemb_cred(gen_cred_id).argmax(dim=1).tolist()
        cred_id = [self.uniq_cred[i] for i in cred_idxs]

        debt_idxs = deemb_debt(gen_debt_id).argmax(dim=1).tolist()
        debt_id = [self.uniq_debt[i] for i in debt_idxs]

        acc_balance = pow(5, gen_log_balance) - 1
        trans_amount = acc_balance*gen_perc_withdrawed

        acc_balance = acc_balance.squeeze().tolist()
        trans_amount = trans_amount.squeeze().tolist()

        trans_types = self.cols_trans_types[gen_trans_types.argmax(dim=1).cpu()].tolist()
        trans_ops = self.cols_trans_ops[gen_trans_ops.argmax(dim=1).cpu()].tolist()
        trans_descs = self.cols_trans_descs[gen_trans_descs.argmax(dim=1).cpu()].tolist()
        bank_code_debtor = self.cols_bank_code_debtor[gen_bank_code_debtor.argmax(dim=1).cpu()].tolist()

        df = pd.DataFrame(list(zip(
            cred_id,
            debt_id,

            acc_balance,
            trans_amount,

            trans_types,
            trans_ops,
            trans_descs,
            bank_code_debtor
        )))

        #df.replace(Trainer.NAN_VAL_TRAIN_STR, None, inplace=True)

        return df