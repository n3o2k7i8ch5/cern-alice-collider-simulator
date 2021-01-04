import os

import matplotlib.pyplot as plt

last_saved_cnt = 0


def show_quality(real, gen, save=False, loss=None, mse_loss=None, kld_loss=None, feature_range=None, title: str=None):
    global last_saved_cnt

    if feature_range is None:
        start_point = 0
        end_point = real.shape[1]
    else:
        feature_count = real.size()[-1]
        start_point = feature_range[0] if feature_range[0] > 0 else feature_count + feature_range[0]
        end_point = feature_range[1] if feature_range[1] > 0 else feature_count + feature_range[1]

    for i in range(start_point, end_point):

        if title is None:
            _title = 'Feature ' + str(i)
        else:
            _title = title + ' - Feature ' + str(i)

        plt.figure(_title)
        plt.title(_title)
        real_attr = real[:, i].detach().cpu().flatten()
        gen_attr = gen[:, i].detach().cpu().flatten()

        plt.hist([real_attr, gen_attr],
                 range=(-5, 5),
                 stacked=False,
                 bins=100,
                 histtype='stepfilled',
                 label=['real', 'fake'],
                 color=['blue', 'red'],
                 alpha=0.5
                 )
        plt.legend()
        if save:
            folder = 'vae_plots'
            if not os.path.isdir(folder):
                os.mkdir(folder)
            loss_str = ''

            if mse_loss is not None:
                loss_str += f'loss: {"{:.3f}".format(round(loss, 3))}'
            if mse_loss is not None:
                loss_str += f' mse_loss: {"{:.3f}".format(round(mse_loss, 3))}'
            if kld_loss is not None:
                loss_str += f' kld_loss: {"{:.3f}".format(round(kld_loss, 3))}'

            plt.savefig(f'{folder}/plot_{i}_{last_saved_cnt}_{loss_str}.png')

        plt.show()
        plt.ion()

    last_saved_cnt += 1
    plt.pause(0.001)


def show_lat_histograms(lat_mean, lat_logvar):
    plt.figure('Latent space quality')

    lat_mean = lat_mean.detach().cpu().flatten()
    lat_logvar = lat_logvar.detach().cpu().flatten()

    plt.hist([lat_mean, lat_logvar],
             range=(-2, 2),
             stacked=False,
             bins=100,
             histtype='stepfilled',
             label=['mean', 'logvar'],
             color=['green', 'orange'],
             alpha=0.5
             )
    plt.legend()
    plt.show()
    plt.ion()
