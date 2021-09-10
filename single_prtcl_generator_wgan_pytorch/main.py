import torch

from common.consts import FEATURES
from i_trainer.load_data import load_data
from single_prtcl_generator_wgan_pytorch.trainer import Trainer

### TRAINING
trainer = Trainer()
#generator, discriminator, embedder, deembedder = trainer.train(epochs=15, load=False)
wgan_gen, wgan_disc = trainer.create_model()
wgan_gen.load_state_dict(torch.load(trainer.GENERATOR_SAVE_PATH))

embedder = trainer.create_embedder()
deembedder = trainer.create_deembedder()
'''
_data = load_data()
_data = _data[:3000, :]
trainer.gen_show_comp_hists(
                        wgan_gen,
                        _data,
                        attr_idxs=[FEATURES - 8, FEATURES - 7, FEATURES - 6, FEATURES - 5],
                        embedders=[embedder],
                        emb=False,
                        sample_cnt=3000,
                        deembedder=deembedder,
                    )
'''
import time

start = time.time()
trainer.gen_autoenc_data(70000, wgan_gen)
print(time.time() - start)