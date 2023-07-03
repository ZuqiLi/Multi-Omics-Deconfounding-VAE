import os 
import numpy as np 
import torch
from scipy import stats


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
    mmd = x_kernel + y_kernel - 2*xy_kernel
    return mmd.sum()


''' 
QC scores
'''

def reconAcc_pearsonCorr(x1, x1_hat, x2, x2_hat):
    ''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
    r2_x1 = []
    for i in range(x1_hat.shape[1]):
        r2_x1.append(stats.pearsonr(x1[:,i], x1_hat[:,i])[0])
    r2_x2 = []
    for i in range(x2_hat.shape[1]):
        r2_x2.append(stats.pearsonr(x2[:,i], x2_hat[:,i])[0])
    return r2_x1, r2_x2

def reconAcc_relativeError(x1, x1_hat, x2, x2_hat):
    ''' Reconstruction accuracy (relative error - L2 norm ratio) '''
    numerator = np.linalg.norm(x1-x1_hat) +  np.linalg.norm(x2 - x2_hat)
    denominator = np.linalg.norm(x1) +  np.linalg.norm(x2)
    relativeError = numerator / denominator
    return relativeError
