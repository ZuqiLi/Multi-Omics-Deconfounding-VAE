import os 
import numpy as np 
import torch


''' 
Regularisation 
'''

def kld(mu, log_var):
    """
    Kullback–Leibler divergence regularizer
    This is the KL of q(z|x) given that the 
    distribution is N(0,1) (or any known distribution)
    """
    kld = 1 + log_var - mu**2- torch.exp(log_var)
    kld = kld.sum(dim = 1)
    kld *= -0.5
    return kld.mean(dim=0)


def mmd(x, y):
    """Maximum Mean Discrepancy regularizer using Gaussian Kernel"""
    def compute_kernel(a, b):
        a_size = a.shape[0]
        b_size = b.shape[0]
        dim = a.shape[1] 
        tiled_a = a.view(a_size, 1, dim).expand(a_size, b_size, dim)
        tiled_b = b.view(1, b_size, dim).expand(a_size, b_size, dim)
        kernel = torch.exp(-(tiled_a - tiled_b).pow(2).mean(2) / dim)
        return kernel

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
