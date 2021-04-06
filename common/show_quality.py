import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import config

last_saved_cnt = 0


def show_quality(real, gen, save=False, loss=None, mse_loss=None, kld_loss=None, feature_range=None, title: str = None):
    global last_saved_cnt

    if feature_range is None:
        start_point = 0
        if len(real.shape) == 1:
            end_point = 1
        else:
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

        real_data = real[:, i].detach().cpu().flatten()
        gen_data = gen[:, i].detach().cpu().flatten()

        show_comp_hists(real_data, gen_data, save_file_name=f'plot_{i}', title=_title)

    last_saved_cnt += 1
    plt.pause(0.001)


def show_lat_histograms(lat_mean, lat_logvar, reprod: bool, save: bool = False, epoch: int = None, batch: int = None):

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
    plt.title('Reproduction histograms' if reprod else 'Generated data histograms')
    plt.legend()
    plt.show()
    plt.ion()
    if save:
        title = 'reprod' if reprod else 'gener'
        plt.savefig(f'{config.plot_folder}/lat_hists_{epoch}_{batch}_{title}.png')

