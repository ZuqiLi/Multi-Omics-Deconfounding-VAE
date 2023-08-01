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
    if reduction == "mean":
        return kld.mean(dim=0)
    if reduction == "sum":
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


def correlation(z, c, nonneg='square'):
    '''
    Regularization for Pearson correlation between every latent vector and the confounder
    '''
    z_ = z - torch.mean(z, 0)
    c_ = c - torch.mean(c)
    num = torch.matmul(z_.T, c_).flatten()
    den = torch.sqrt(torch.sum(z_**2, 0)) * torch.sqrt(torch.sum(c_**2))
    corr = num / (den + 1e-8)

    if nonneg == 'square':
        corr = corr ** 2
    if nonneg == 'abs':
        corr = corr.abs()
    corr = corr.sum()

    return corr


def MI_with_hist(x, y, n_bins=50):
    '''
    compute the mutual information between 2 1D tensors of the same shape
    '''
    epsilon = 1e-10

    # compute marginal and joint probability distributions
    pdf_xy = torch.histogramdd(torch.column_stack((x, y)), n_bins)[0]
    pdf_x = torch.histogram(x, n_bins)[0]
    pdf_y = torch.histogram(y, n_bins)[0]
    pdf_xy = nn.Parameter(pdf_xy, requires_grad=False)
    pdf_x = nn.Parameter(pdf_x, requires_grad=False)
    pdf_y = nn.Parameter(pdf_y, requires_grad=False)

    # normalize PDF
    pdf_xy = pdf_xy / (torch.sum(pdf_xy) + epsilon)
    pdf_x = pdf_x / (torch.sum(pdf_x) + epsilon)
    pdf_y = pdf_y / (torch.sum(pdf_y) + epsilon)

    # compute entropy
    H_xy = -torch.sum(pdf_xy * torch.log2(pdf_xy + epsilon))
    H_x = -torch.sum(pdf_x * torch.log2(pdf_x + epsilon))
    H_y = -torch.sum(pdf_y * torch.log2(pdf_y + epsilon))

    # compute MI
    MI = H_x + H_y - H_xy
    # normalize MI
    MI = 2 * MI / (H_x + H_y)

    return MI


def MI_with_KDE(x, y, n_bins=10, bw=0.1, epsilon=1e-10):
    '''
    Compute the mutual information between 2 1D tensors of the same shape.
    Inspired by Kornia AI and https://github.com/connorlee77/pytorch-mutual-information/
    '''

    def marginal_pdf(values, bins, bw=0.01, epsilon=1e-10):
        n = values.shape[0]
        pi = torch.tensor(3.14159265359)
        bw = (4/3)**0.2 * torch.std(values) * n**(-0.2)

        residuals = values.unsqueeze(1) - bins.unsqueeze(0)
        kernel_values = (1/bw/torch.sqrt(2*pi)) * torch.exp(-0.5*(residuals/bw).pow(2))

        pdf = torch.mean(kernel_values, dim=0)
        normalization = torch.sum(pdf) + epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def joint_pdf(kernel_values1, kernel_values2, epsilon=1e-10):
        joint_kernel_values = torch.matmul(kernel_values1.T, kernel_values2)
        normalization = torch.sum(joint_kernel_values) + epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    x_bins = torch.linspace(x.min().item(), x.max().item(), n_bins, device=device)
    y_bins = torch.linspace(y.min().item(), y.max().item(), n_bins, device=device)

    # compute marginal and joint probability distributions
    pdf_x, kernel_values1 = marginal_pdf(x, x_bins)
    #y = y.repeat(1, x.shape[1])
    pdf_y, kernel_values2 = marginal_pdf(y, y_bins)
    pdf_xy = joint_pdf(kernel_values1, kernel_values2)

    # compute entropy
    H_x = -torch.sum(pdf_x * torch.log(pdf_x + epsilon))
    H_y = -torch.sum(pdf_y * torch.log(pdf_y + epsilon))
    H_xy = -torch.sum(pdf_xy * torch.log(pdf_xy + epsilon))

    # compute MI
    MI = H_x + H_y - H_xy
    # normalize MI
    MI = 2 * MI / (H_x + H_y)

    return MI


def mutualInfo(z, c, method='kde'):
    '''
    Regularization for mutual information between every latent vector and the confounder
    '''
    MI = []
    for i in range(z.shape[1]):
        if method == 'kde':
            MI.append(MI_with_KDE(z[:,i], torch.flatten(c)))
        if method == 'hist':
            MI.append(MI_with_hist(z[:,i], torch.flatten(c)))
    MI = torch.stack(MI).sum()

    return MI


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
