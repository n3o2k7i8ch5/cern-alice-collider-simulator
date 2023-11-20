import torch


def sample_norm(mean, logvar):

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std