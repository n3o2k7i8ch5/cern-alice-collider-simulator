import matplotlib.pyplot as plt
import numpy as np
import torch

from _legacy.common.consts import FEATURES
from _legacy.i_trainer.load_data import load_data
from _legacy.single_prtcl_generator_wgan_pytorch.trainer import Trainer


### TRAINING
trainer = Trainer()
#generator, discriminator, embedder, deembedder = trainer.train(epochs=15, load=False)
wgan_gen, wgan_disc = trainer.create_model()
wgan_gen.load_state_dict(torch.load(trainer.GENERATOR_SAVE_PATH))

embedder = trainer.create_embedder()
deembedder = trainer.create_deembedder()

_data = load_data()

_data = _data[:10000, :]


trainer.gen_show_comp_hists(
                        wgan_gen,
                        _data,
                        attr_idxs=[7-4, 8-4, 9-4, FEATURES - 5],
                        embedders=[embedder],
                        emb=False,
                        sample_cnt=1000,
                        deembedder=deembedder,
                    )


'''
import time

start = time.time()
trainer.gen_autoenc_data(70000, wgan_gen)
print(time.time() - start)
'''

prtcl_cnt = 5000

gen_data_1 = trainer.gen_autoenc_data(prtcl_cnt, wgan_gen)
gen_data_1 = deembedder.deemb(gen_data_1).cpu().numpy()[:, 7-4]
real_data_1 = _data.numpy()[:prtcl_cnt, 7-4]

gen_data_2 = trainer.gen_autoenc_data(prtcl_cnt, wgan_gen)
gen_data_2 = deembedder.deemb(gen_data_2).cpu().numpy()[:, 8-4]
real_data_2 = _data.numpy()[:prtcl_cnt, 8-4]

gen_data_3 = trainer.gen_autoenc_data(prtcl_cnt, wgan_gen)
gen_data_3 = deembedder.deemb(gen_data_3).cpu().numpy()[:, 9-4]
real_data_3 = _data.numpy()[:prtcl_cnt, 9-4]

'''
print('mean gen', mean(gen_data_1[gen_data_1 > .5]))
print('std gen', std(gen_data_1[gen_data_1 > .5]))
print('min gen', min(gen_data_1))
print('max gen', max(gen_data_1))

print('\nmean real', mean(real_data_1[real_data_1 > .5]))
print('std real', std(real_data_1[real_data_1 > .5]))
print('min real', min(real_data_1))
print('max real', max(real_data_1))
'''

real_sums = []
gen_sums = []

for i in range(prtcl_cnt):

    real = np.sum(real_data_1[:i])# + np.sum(real_data_2[:i]) + np.sum(real_data_3[:i])
    gen = np.sum(gen_data_1[:i])# + np.sum(gen_data_2[:i]) + np.sum(gen_data_3[:i])

    real_sums.append(real)
    gen_sums.append(gen)

#print('sum real', np.sum(real_data[:, 3]))
#print('sum gen', np.sum(gen_data[:, 3]))

#print(real_sums)

plt.plot(real_sums)
plt.plot(gen_sums)

plt.legend(['Pythia8', 'WGAN'])
plt.ylabel('Suma pędu cząstek')
plt.xlabel('Liczba sumowanych cząstek')
plt.title('Porównanie zasady zachowania pędu dla generowanych\ncząstek między symulatorem Pythia8 i WGAN #4', size=14)

plt.show()

#plt.hist(gen_data)
#plt.show()