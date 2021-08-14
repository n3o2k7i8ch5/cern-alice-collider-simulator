import pickle
import matplotlib.pyplot as plt


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

plt.plot(_errs_gan_lrelu)
plt.plot(_errs_gan_tanh)
plt.legend(['tanh', 'leaky ReLU'])
plt.show()

'''
plt.plot(_errs_gan)
plt.plot(_errs_wgan)
plt.plot(_errs_vae)
plt.legend(['gan', 'wgan', 'vae'])

plt.show()

'''