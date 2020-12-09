import matplotlib.pyplot as plt
import torch

from common.consts import *


def show_plots(real_data: torch.Tensor, fake_data: torch.Tensor, to_show_end_cnt=None):
    real_data = real_data.detach().cpu()
    fake_data = fake_data.detach().cpu()

    IMG_SAMPLES_TO_SHOW = 80

    # '''
    plt.figure()
    plt.title('Real img')
    if len(real_data.size()) > 2:
        plt.imshow(real_data[0, :].split(split_size=IMG_SAMPLES_TO_SHOW, dim=0)[0])
    else:
        plt.imshow(real_data.split(split_size=IMG_SAMPLES_TO_SHOW, dim=0)[0])
    plt.colorbar()
    plt.ion()
    plt.pause(0.001)
    # '''

    # '''
    plt.figure()
    plt.title('Fake img')
    if len(fake_data.size()) > 2:
        plt.imshow(fake_data[0, :].split(split_size=IMG_SAMPLES_TO_SHOW, dim=0)[0])
    else:
        fake_data_to_show = fake_data.split(split_size=IMG_SAMPLES_TO_SHOW, dim=0)[0]
        plt.imshow(fake_data_to_show)

    plt.colorbar()
    plt.ion()
    plt.pause(0.001)
    # '''

    feature_count = real_data.size()[-1]

    start_point = feature_count - to_show_end_cnt if to_show_end_cnt is not None else 0

    if start_point<0:
        start_point = 0

    if len(real_data.size()) > 2:
        for h_lay_idx in range(start_point, feature_count):
            plt.figure()
            plt.title('Real hist')

            plt.hist(real_data[:, :, h_lay_idx].flatten().numpy(), bins=20, alpha=0.5, color='b', label='real', density=True)
            plt.hist(fake_data[:, :, h_lay_idx].flatten().numpy(), bins=20, alpha=0.5, color='r', label='gen', density=True)

            plt.legend(prop={'size': 10})
            plt.show()
    else:
        for h_lay_idx in range(start_point, feature_count):
            plt.figure()
            plt.title('Real hist')

            _real = real_data[:, h_lay_idx].flatten().numpy()
            plt.hist(_real, bins=50, alpha=0.5, color='b', label='real', density=True)
            plt.hist(fake_data[:, h_lay_idx].flatten().numpy(), bins=50, alpha=0.5, color='r', label='gen',
                     density=True)

            plt.legend(prop={'size': 10})
            plt.show()

    plt.pause(0.001)
