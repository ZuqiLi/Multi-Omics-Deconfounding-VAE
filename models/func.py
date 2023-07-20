import os 
import numpy as np 
import torch
from scipy import stats
import torch.nn as nn

''' 
Weight initialisation 
'''
def init_weights(layer, method="rai"):
    ''' Initialise layers for smoother training '''    
    def RAI_(weight, bias) -> torch.Tensor:
        def RAI_lu_2019(fan_in, fan_out):
            """Randomized asymmetric initializer.
            It draws samples using RAI where fan_in is the number of input units in the weight
            tensor and fan_out is the number of output units in the weight tensor.

            Lu, Lu, et al. "Dying relu and initialization: Theory and numerical examples." arXiv preprint arXiv:1903.06733 (2019).
            """
            V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
            for j in range(fan_out):
                k = np.random.randint(0, high=fan_in + 1)
                V[j, k] = np.random.beta(2, 1)
                W = V[:, :-1].T
                b = V[:, -1]
            return W.astype(np.float32), b.astype(np.float32)
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        w, b = RAI_lu_2019(fan_in, fan_out)
        with torch.no_grad():
            return torch.tensor(w), torch.tensor(b)
                
    if isinstance(layer, nn.Linear):
        if method == "rai":
            RAI_(layer.weight.data, layer.bias.data)
        if method == "he":
            torch.nn.init.kaiming_normal_(layer.weight.data)
        if method == "xavier":
            torch.nn.init.xavier_normal_(layer.weight.data)


''' 
Regularisation 
'''

def kld(mu, log_var, reduction="sum"):
    """
    Kullback–Leibler divergence regularizer
    This is the KL of q(z|x) given that the 
    distribution is N(0,1) (or any known distribution)
    """
    kld = 1 + log_var - mu**2- torch.exp(log_var)
    kld = kld.sum(dim = 1)
    kld *= -0.5
    if reduction is "mean":
        return kld.mean(dim=0)
    if reduction is "sum":
        return kld.sum(dim=0)


def mmd(x, y, reduction='sum'):
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

    if reduction == 'mean':
        return x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    if reduction == 'sum':
        return x_kernel.sum() + y_kernel.sum() - 2*xy_kernel.sum()


def mse(pred, true):
    ''' MSE loss for regression '''
    loss = torch.nn.MSELoss(reduction="sum")
    mse_loss = loss(torch.flatten(pred), true)
    return mse_loss

def bce(pred, true):
    ''' BCE loss for clf '''
    loss = torch.nn.BCELoss(reduction="sum")
    bce_loss = loss(pred, true)
    return bce_loss

def crossEntropy(pred, true):
    ''' Crossentropy loss for clf '''
    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    crossent = loss(pred, true)
    return crossent

''' 
QC scores
'''

def reconAcc_pearsonCorr(x1, x1_hat):
    ''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
    r2_x1 = []
    for i in range(x1_hat.shape[1]):
        r2_x1.append(stats.pearsonr(x1[:,i], x1_hat[:,i])[0])
    return r2_x1

def reconAcc_relativeError(x1, x1_hat, x2, x2_hat):
    ''' Reconstruction accuracy (relative error - L2 norm ratio) '''
    error_x1 = np.linalg.norm(x1 - x1_hat)
    error_x2 = np.linalg.norm(x2 - x2_hat)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    RE_x1 = error_x1 / norm_x1
    RE_x2 = error_x2 / norm_x2
    RE_x1x2 = (error_x1 + error_x2) / (norm_x1 + norm_x2)
    return RE_x1, RE_x2, RE_x1x2
