import pickle
from math import log

import matplotlib.pyplot as plt
import numpy as np


def mean_window(data, win_size):
    _data = []

    for i in range(len(data) - win_size):
        _sum = 0
        for j in range(win_size):
            _sum += data[i + j]

        _data.append(_sum/win_size)

    return _data


def read_from_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


_win = 10

errs_gan_tanh = read_from_pickle('data/errs_gan_tanh.model')
errs_wgan_tanh = read_from_pickle('data/errs_wgan_tanh.model')
errs_vae_tanh = read_from_pickle('data/errs_vae_tanh.model')

errs_gan_lrelu = read_from_pickle('data/errs_gan_lrelu.model')
errs_wgan_lrelu = read_from_pickle('data/errs_wgan_lrelu.model')
errs_vae_lrelu = read_from_pickle('data/errs_vae_lrelu.model')

#GAN
_errs_gan_tanh = mean_window(errs_gan_tanh[1], win_size=_win)
_errs_gan_tanh = [_errs_gan_tanh[i] for i in range(11*15)]

_errs_gan_lrelu = mean_window(errs_gan_lrelu[1], win_size=_win)
_errs_gan_lrelu = [_errs_gan_lrelu[i] for i in range(11*15)]

#WGAN
_errs_wgan_lrelu = mean_window(errs_wgan_lrelu[1], win_size=_win)
_errs_wgan_lrelu = [_errs_wgan_lrelu[i] for i in range(11*15)]

_errs_wgan_tanh = mean_window(errs_wgan_tanh[1], win_size=_win)
_errs_wgan_tanh = [_errs_wgan_tanh[i] for i in range(11*15)]

#VAE
_errs_vae_tanh = mean_window(errs_vae_tanh[1], win_size=_win)
_errs_vae_tanh = [_errs_vae_tanh[i] for i in range(11*15)]

_errs_vae_lrelu = mean_window(errs_vae_lrelu[1], win_size=_win)
_errs_vae_lrelu = [_errs_vae_lrelu[i] for i in range(11*15)]

'''
plt.plot(_errs_wgan_tanh)
plt.plot(_errs_wgan_lrelu)
plt.title('Jakość generowanych danych\nw funkcji czasu uczenia modelu.', size=14)
plt.legend(['tanh', 'leaky ReLU'])
plt.xlabel('Etap uczenia mierzony w batch-ach.')
plt.ylabel('Różnica rozkładu histogramu danych rzecz. i syntet.\n(mierzonej funkcją Wassersteina).')

plt.show()


plt.plot(_errs_gan_lrelu)
plt.plot(_errs_wgan_lrelu)
plt.plot(_errs_vae_lrelu)
plt.legend(['gan', 'wgan', 'vae'])

plt.show()
'''


### Czy małe zbioru są ok?
'''
size = [50, 100, 500, 1000, 1500, 2000, 3000, 5000, 10_000, 50_000]
std = [0.6565979, 0.764541617, 0.815992631, 0.910672786, 0.828401661, 0.859607306, 0.884218504, 0.859313956, 0.902838184, 0.906131092]

std_1 = [1.175471628, 1.686083556, 6,172178295, 11.8957253, 19.49050209, 25.32687575, 37.50129564, 64.72562194, 127.9618696, 674.9296077]
std_2 = [0.771812202, 1.289081048, 5.036452008, 10.8331133, 16.1459643, 21.77116744, 33.15933954, 55.61963022, 115.528862, 611.5747024]



plt.plot(
    size,
    std
)
plt.xscale('log')

plt.title('Zależność wariancji histogramu danych syntetycznych\ndo wariancji histogramu danych rzeczywistych', size=14)
plt.xlabel('Liczba generowanych przykładów')
plt.ylabel('stosunek wariancji hist(dane syntetyczne)\ndo wariancji hist(dane rzeczywiste)')

plt.show()
'''



### TIME
'''
sample_cnt = [30, 100, 300, 1000, 3000, 10000, 30000, 70000, 100_000, 1_000_000, 10_000_000]
gan = [0.00130295753479004, 0.00143170356750488, 0.00161838531494141, 0.00209736824035645, 0.00298571586608887, 0.00574827194213867, 0.0131165981292725, 0.0274009704589844, 0.0574827194213867, 0.564827194213867, 5.75827194213867]
wgan = [0.00117325782775879, 0.00206232070922852, 0.00175213813781738, 0.00190877914428711, 0.00317859649658203, 0.0055544376373291, 0.012000560760498, 0.026212215423584, 0.055544376373291, 0.56544376373291, 5.9544376373291]
vae = [0.00111579895019531, 0.00140190124511719, 0.0013725757598877, 0.0017096996307373, 0.00299406051635742, 0.00580120086669922, 0.0128393173217773, 0.025198221206665, 0.0580120086669922, 0.590120086669922, 5.92120086669922]
pythia = [0.631861, 0.589083, 0.584402, 0.586129, 0.642659, 0.747605, 0.901462, 1.145348, 1.297201, 5.967705, 62.852554]

plt.plot(sample_cnt, pythia)
plt.plot(sample_cnt, vae)
plt.plot(sample_cnt, gan)
plt.plot(sample_cnt, wgan)
plt.xscale('log')
plt.yscale('log')

plt.title('Zależność czasu generacji danych syntetycznych\nod liczby generowanych przykładów', size=14)
plt.xlabel('Liczba generowanych przykładów')
plt.ylabel('Czas trwania generacji danych')
plt.legend(['pythia8', 'gan', 'wgan', 'vae'])
plt.show()
print(len(gan), len(sample_cnt))
'''

