# This is a copy of the X-VAE architecture below as a starting point of our Multi-view Deconfounding VAE
# Source: https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
# Paper: https://www.frontiersin.org/articles/10.3389/fgene.2019.01205

from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from keras.models import Model

from models.common import sse, bce, mmd, sampling, kl_regu
from keras.losses import mean_squared_error,binary_crossentropy
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed


def sse(true, pred):
    return K.sum(K.square(true - pred), axis=1)

def cce(true, pred):
    return K.mean(K.sparse_categorical_crossentropy(true, pred, from_logits=True), axis=1)

def bce(true, pred):
    return K.sum(K.binary_crossentropy(true, pred, from_logits=True), axis=1)

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

def kl_regu(z_mean,z_log_sigma):
    #regularizer. this is the KL of q(z|x) given that the 
    #distribution is N(0,1) (or any known distribution)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss
  
    
class XVAE:
    def __init__(self, args):
        self.args = args
        self.vae = None
        self.encoder = None

    def build_model(self):
        
        np.random.seed(42)
        set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        s1_inp = Input(shape=(self.args.s1_input_size,))
        s2_inp = Input(shape=(self.args.s2_input_size,))
        inputs = [s1_inp, s2_inp]

        # ------------ Concat Layer -----------------
        x1 = Dense(self.args.ds, activation=self.args.act)(s1_inp)
        x1 = BN()(x1)

        x2 = Dense(self.args.ds, activation=self.args.act)(s2_inp)
        x2 = BN()(x2)

        x = Concatenate(axis=-1)([x1, x2])

        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ls, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)
        
        x=Dropout(self.args.dropout)(x)
        # ------------ Dense branches ------------
        x1 = Dense(self.args.ds, activation=self.args.act)(x)
        x1 = BN()(x1)
        x2 = Dense(self.args.ds, activation=self.args.act)(x)
        x2 = BN()(x2)

        # ------------ Out -----------------------
        s1_out = Dense(self.args.s1_input_size, activation='sigmoid')(x1)
        
        if self.args.integration == 'Clin+CNA':
            s2_out = Dense(self.args.s2_input_size,activation='sigmoid')(x2)
        else:
            s2_out = Dense(self.args.s2_input_size)(x2)

        decoder = Model(latent_inputs, [s1_out, s2_out], name='decoder')
        decoder.summary()

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_x')

        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ls]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)
          
        
        
        s1_loss= binary_crossentropy(inputs[0], outputs[0])

        if self.args.integration == 'Clin+CNA':
            s2_loss =binary_crossentropy(inputs[1], outputs[1])
        else:
            s2_loss =mean_squared_error(inputs[1], outputs[1])
        
        
        reconstruction_loss = s1_loss+s2_loss
        vae_loss = K.mean(reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False, decay=0.001)
        self.vae.compile(optimizer=adam, metrics=[mean_squared_error, mean_squared_error])
        self.vae.summary()

    def train(self, s1_train, s2_train, s1_test, s2_test):
        self.vae.fit([s1_train, s2_train], epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True,
                     validation_data=([s1_test, s2_test], None))
        if self.args.save_model:
            self.vae.save_weights('./models/vae_xvae.h5')

    def predict(self, s1_data, s2_data):
        return self.encoder.predict([s1_data, s2_data], batch_size=self.args.bs)[0]
