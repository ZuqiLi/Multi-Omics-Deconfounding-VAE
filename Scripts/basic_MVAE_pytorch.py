# This is the PyTorch Lightning version of the X-VAE implementation
# Source: https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
# Paper: https://www.frontiersin.org/articles/10.3389/fgene.2019.01205
# PyTorch Lightning example: https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-with-pytorch-lightning-13dbc559ba4b


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
from pytorch_lightning import Trainer
import numpy as np


def compute_kernel(x, y):
    x_size = K.shape(x)[0] #K.shape(x)[0] #need to fix to get batch size
    y_size = K.shape(y)[0]
    dim = K.int_shape(x)[1] #K.get_shape(x)[1] #x.get_shape().as_list()[1]   
    tiled_x = K.tile(K.reshape(x,K.stack([x_size, 1, dim])), K.stack([1,y_size, 1]))
    tiled_y = K.tile(K.reshape(y,K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
    kernel_input = K.exp(-K.mean((tiled_x - tiled_y)**2, axis=2)) / K.cast(dim, np.float32)
    return kernel_input

def mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2*K.mean(xy_kernel)
    return mmd

def kld(mu, log_var):
    #regularizer. this is the KL of q(z|x) given that the 
    #distribution is N(0,1) (or any known distribution)
    kld = 1 + log_var - mu**2- torch.exp(log_var)
    kld = kld.sum(dim = 1)
    kld *= -0.5
    return kld.mean(dim=0)

    
class XVAE(L.LightningModule):
    def __init__(self, x1_size, x2_size, ls, distance, bs, beta, save_model):
        super().__init__()
        self.ls = ls # latent space
        self.distance = distance
        self.bs = bs # batch size
        self.beta = beta # weight for distance term in loss function
        self.save_model = save_model # true or false
        
        # encoder
        self.encoder.x1_fc = nn.Linear(x1_size, 128)
        self.encoder.x2_fc = nn.Linear(x2_size, 128)
        self.encoder.fuse = nn.Linear(128+128, 128)
        # embedding
        self.embed.mu = nn.Linear(128, ls)
        self.embed.log_var = nn.Linear(128, ls)
        # decoder
        self.decoder.sample = nn.Linear(ls, 128)
        self.decoder.x1_fc = nn.Linear(128, x1_size)
        self.decoder.x2_fc = nn.Linear(128, x2_size)
        
    def sample_z(self, mu, log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z= z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z
    
    def encode(self, x1, x2):
        x1 = F.relu(self.encoder.x1_fc(x1))
        x2 = F.relu(self.encoder.x2_fc(x2))

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.encoder.fuse(x))

        mu = self.embed.mu(x)
        log_var = self.embed.log_var(x)
        return mu, log_var
    
    def decode(self, z):
        x_hat = F.relu(self.decoder.sample(z))
        x1_hat = F.relu(self.decoder.x1_fc(x_hat))
        x2_hat = F.relu(self.decoder.x2_fc(x_hat))
        return x1_hat, x2_hat
        
    def forward(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = decode(z)
        return x1_hat, x2_hat
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay=0.001)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def compute_loss(self, batch):
        x1, x2 = batch
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = decode(z)

        if self.distance == "mmd":
            true_samples = torch.randn([self.bs, self.ls])
            distance = mmd(true_samples, z)
        if self.distance == "kld":
            distance = kld(mu, log_var)         
    
        recon_loss_criterion = nn.MSELoss(reduction="mean")
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss_x2 = recon_loss_criterion(x2, x2_hat)
        recon_loss = recon_loss_x1 + recon_loss_x2
        
        vae_loss = recon_loss + self.beta * distance
        return vae_loss
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss , on_step = True, on_epoch = True)
        return x_out,loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss , on_step = True, on_epoch = True)
        return x_out,loss

   
def main():
    # Initializing Dataloader
    train_set = MNIST('data/',download = True,train = True,transform=data_transform)
    train_loader = Dataloader(train_set,batch_size=64)
    val_set = MNIST('data/',download = True,train = False,transform=data_transform)
    val_loader = Dataloader(val_set,batch_size=64)

    model = XVAE(x1_size, x2_size, ls, distance, bs, beta, save_model)
    # Initializing Trainer and setting parameters
    trainer = Trainer(accelerator="auto", devices=1, max_epochs=25)
    # Using trainer to fit vae model to dataset
    trainer.fit(model, train_loader, val_loader)
    
    # Test best model on validation and test set
    #val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    #test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    
    
if __name__ == "__main__":
    main()
    
    
    
