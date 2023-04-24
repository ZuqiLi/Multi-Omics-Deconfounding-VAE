import os 
import numpy as np 
import torch


''' 
Regularisation 
'''

def kld(mu, log_var):
    #regularizer. this is the KL of q(z|x) given that the 
    #distribution is N(0,1) (or any known distribution)
    kld = 1 + log_var - mu**2- torch.exp(log_var)
    kld = kld.sum(dim = 1)
    kld *= -0.5
    return kld.mean(dim=0)


### Still have to translate to pytorch 
### from keras import backend as K
# def mmd(x, y):
#     def compute_kernel(a,b):
#         a_size = K.shape(a)[0] #K.shape(a)[0] #need to fia to get batch size
#         b_size = K.shape(b)[0]
#         dim = K.int_shape(a)[1] #K.get_shape(a)[1] #a.get_shape().as_list()[1]   
#         tiled_a = K.tile(K.reshape(a,K.stack([a_size, 1, dim])), K.stack([1,b_size, 1]))
#         tiled_b = K.tile(K.reshape(b,K.stack([1, b_size, dim])), K.stack([a_size, 1, 1]))
#         kernel_input = K.exp(-K.mean((tiled_a - tiled_b)**2, axis=2)) / K.cast(dim, np.float32)
#         return kernel_input
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2*K.mean(xy_kernel)
#     return mmd