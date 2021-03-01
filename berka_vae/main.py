from berka_vae.consts import CRED_EMBED_DIM, DEBT_EMBED_DIM
from berka_vae.trainer import Trainer
from common.show_quality import show_comp_hists
from single_prtcl_generator_vae_pytorch.models.pdg_deembedder import PDGDeembedder

import pandas as pd

trainer = Trainer()
# vae, embedder_cred, embedder_debt, deembedder_cred, deembedder_debt = trainer.train(epochs=5, load=True)

trainer.load_trans_data()
vae = trainer.create_vae(load=True)
deemb_cred = PDGDeembedder(pdg_count=20, pdg_embed_dim=CRED_EMBED_DIM, device=trainer.device)
deemb_debt = PDGDeembedder(pdg_count=20, pdg_embed_dim=DEBT_EMBED_DIM, device=trainer.device)

real_df = pd.read_csv('berka_trans_mapped.csv', sep='\t')
gen_data = trainer.gen_autoenc_data(1024 * 16, vae)
gen_df = trainer.convert_to_dataframe(gen_data, deemb_cred, deemb_debt)

real_col = real_df['trans_amount'][:1024 * 16]

gen_col = gen_df[2]
show_comp_hists(
    real_col.to_numpy(),
    gen_col.to_numpy(),
    title='Trans amount histogram',
    xlabel='Trans. amount',
    ylabel='Num. of trans.'
)

real_col = real_df['acc_balance'][:1024 * 16]
gen_col = gen_df[3]
show_comp_hists(
    real_col.to_numpy(),
    gen_col.to_numpy(),
    title='Account balance histogram',
    xlabel='Trans. amount',
    ylabel='Num. of trans.'
)

real_col = real_df['trans_type'][:1024 * 16]
gen_col = gen_df[4]
show_comp_hists(real_col.to_numpy(), gen_col.to_numpy(), title='Trans type')

real_col = real_df['trans_operation'][:1024 * 16]
gen_col = gen_df[5]
show_comp_hists(real_col.to_numpy(), gen_col.to_numpy(), title='Trans operation')

import seaborn as sns
import matplotlib.pyplot as plt


def show_correlations(data, title='Pearson Correlation of Features'):
    colormap = plt.cm.RdBu
    plt.title(title, y=1)
    sns.heatmap(data.astype(float).corr(),
                linewidths=0.1,
                vmax=1.0,
                square=True,
                cmap=colormap,
                linecolor='white',
                annot=True)

    plt.show()


show_correlations(real_df[['trans_amount', 'acc_balance', 'acc_id_creditor', 'acc_id_debtor']], title='real data')
show_correlations(gen_df[[3, 2, 1, 0]], title='gen data')
