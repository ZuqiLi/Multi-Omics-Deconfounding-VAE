# This is a copy of the X-VAE architecture below as a starting point of our Multi-view Deconfounding VAE
# Source: https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
# Paper: https://www.frontiersin.org/articles/10.3389/fgene.2019.01205

from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from keras.models import Model

from keras.losses import mean_squared_error,binary_crossentropy
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed


def compute_kernel(x, y):
    x_size = K.shape(x)[0] #K.shape(x)[0] #need to fix to get batch size
    y_size = K.shape(y)[0]
    dim = K.int_shape(x)[1] #K.get_shape(x)[1] #x.get_shape().as_list()[1]   
    tiled_x = K.tile(K.reshape(x,K.stack([x_size, 1, dim])), K.stack([1,y_size, 1]))
    tiled_y = K.tile(K.reshape(y,K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
    kernel_input = K.exp(-K.mean((tiled_x - tiled_y)**2, axis=2)) / K.cast(dim, tf.float32)
    return kernel_input

def mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2*K.mean(xy_kernel)
    return mmd

def kl_regu(z_mean,z_log_sigma):
    #regularizer. this is the KL of q(z|x) given that the 
    #distribution is N(0,1) (or any known distribution)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss
    
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
  
    
class XVAE:
    def __init__(self, v1_size, v2_size, activation, ls, dropout, distance, bs, beta, epochs, save_model):
        self.v1_size = v1_size
        self.v2_size = v2_size
        self.activation = activation
        self.ls = ls # size of latent space
        self.dropout = dropout
        self.distance = distance
        self.bs = bs # batch size
        self.epochs = epochs
        self.beta = beta # weight for distance term in loss function
        self.save_model = save_model # true or false
        self.vae = None
        self.encoder = None

    def build_model(self):
        
        np.random.seed(42)
        set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        v1_input = Input(shape=(self.v1_size,))
        v2_input = Input(shape=(self.v2_size,))
        inputs = [v1_input, v2_input]

        # ------------ Concat Layer -----------------
        x1 = Dense(256, activation=self.activation)(v1_input)
        x1 = BN()(x1)

        x2 = Dense(256, activation=self.activation)(v2_input)
        x2 = BN()(x2)

        x = Concatenate(axis=-1)([x1, x2])

        x = Dense(256, activation=self.activation)(x)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.ls, name='z_mean')(x)
        z_log_sigma = Dense(self.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(256, activation=self.activation)(x)
        x = BN()(x)
        
        x = Dropout(self.dropout)(x)
        # ------------ Dense branches ------------
        x1 = Dense(256, activation=self.activation)(x)
        x1 = BN()(x1)
        x2 = Dense(256, activation=self.activation)(x)
        x2 = BN()(x2)

        # ------------ Out -----------------------
        v1_out = Dense(self.v1_size, activation='sigmoid')(x1)
        v2_out = Dense(self.v2_size, activation='sigmoid')(x2)

        decoder = Model(latent_inputs, [v1_out, v2_out], name='decoder')
        decoder.summary()

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        # ------------ Loss function -----------------------
        if self.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.bs, self.ls]))
            distance = mmd(true_samples, z)
        if self.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)
          
        v1_loss= binary_crossentropy(inputs[0], outputs[0])
        v2_loss =binary_crossentropy(inputs[1], outputs[1])
        reconstruction_loss = v1_loss + v2_loss
        
        vae_loss = K.mean(reconstruction_loss + self.beta * distance)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False, decay=0.001)
        self.vae.compile(optimizer=adam, metrics=[mean_squared_error, mean_squared_error])
        self.vae.summary()

    def train(self, v1_train, v2_train, v1_test, v2_test):
        self.vae.fit([v1_train, v2_train], epochs=self.epochs, batch_size=self.bs, shuffle=True,
                     validation_data=([v1_test, v2_test], None))
        if self.save_model:
            self.vae.save_weights('./models/vae_xvae.h5')

    def predict(self, v1_data, v2_data):
        return self.encoder.predict([v1_data, v2_data], batch_size=self.bs)[0]
