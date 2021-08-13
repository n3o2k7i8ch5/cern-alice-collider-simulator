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

errs_gan = read_from_pickle('data/errs_gan_lrelu.model')
errs_wgan = read_from_pickle('data/errs_wgan_lrelu.model')
errs_vae = read_from_pickle('data/errs_vae_lrelu.model')

_errs_gan = mean_window(errs_gan[1], win_size=_win)
_errs_gan = [_errs_gan[i] for i in range(11*15)]

_errs_wgan = mean_window(errs_wgan[1], win_size=_win)
_errs_wgan = [_errs_wgan[i] for i in range(11*15)]

_errs_vae = mean_window(errs_vae[1], win_size=_win)
_errs_vae = [_errs_vae[i] for i in range(11*15)]

plt.plot(_errs_gan)
plt.plot(_errs_wgan)
plt.plot(_errs_vae)
plt.legend(['gan', 'wgan', 'vae'])

plt.show()
