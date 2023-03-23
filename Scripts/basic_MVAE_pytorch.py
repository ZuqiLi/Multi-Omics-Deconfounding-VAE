# This is the PyTorch Lightning version of the X-VAE implementation
# Source: https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
# Paper: https://www.frontiersin.org/articles/10.3389/fgene.2019.01205
# PyTorch Lightning example: https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-with-pytorch-lightning-13dbc559ba4b


import os
import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import lightning as L


path = '/usr/local/micapollo01/MIC/DATA/STAFF/zli1/MVDVAE/'

np.random.seed(1234)
torch.manual_seed(1234)
L.seed_everything(1234)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


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
    def __init__(self, x1_size, x2_size, ls, distance, beta, save_model):
        super().__init__()
        self.ls = ls # latent space
        self.distance = distance
        self.beta = beta # weight for distance term in loss function
        self.save_model = save_model # true or false
        
        # encoder
        self.encoder_x1_fc = nn.Linear(x1_size, 128)
        self.encoder_x2_fc = nn.Linear(x2_size, 128)
        self.encoder_fuse = nn.Linear(128+128, 128)
        # embedding
        self.embed_mu = nn.Linear(128, self.ls)
        self.embed_log_var = nn.Linear(128, self.ls)
        # decoder
        self.decoder_sample = nn.Linear(self.ls, 128)
        self.decoder_x1_fc = nn.Linear(128, x1_size)
        self.decoder_x2_fc = nn.Linear(128, x2_size)
        
    def sample_z(self, mu, log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z= z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z
    
    def encode(self, x1, x2):
        x1 = F.relu(self.encoder_x1_fc(x1))
        x2 = F.relu(self.encoder_x2_fc(x2))

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.encoder_fuse(x))

        mu = self.embed_mu(x)
        log_var = self.embed_log_var(x)
        return mu, log_var
    
    def decode(self, z):
        x_hat = F.relu(self.decoder_sample(z))
        x1_hat = F.relu(self.decoder_x1_fc(x_hat))
        x2_hat = F.relu(self.decoder_x2_fc(x_hat))
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
        x1_hat, x2_hat = self.decode(z)

        if self.distance == "mmd":
            true_samples = torch.randn([x1.shape[0], self.ls])
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
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss , on_step = True, on_epoch = True)
        return loss

    
class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
   

def main():
    
    # Load the 2 input datasets
    #X1 = np.loadtxt(path + 'TCGA_mRNAs.csv', delimiter=",", skiprows=1)
    #X2 = np.loadtxt(path + 'TCGA_miRNAs.csv', delimiter=",", skiprows=1)
    # Normalize datasets
    #X1 = normalize(X1, axis=0)
    #X2 = normalize(X2, axis=0)
    # Select the most variable features
    #X1_var = np.argsort(np.var(X1, axis=0))[::-1]
    #X1 = X1[:, X1_var[:2000]]
    #X2_var = np.argsort(np.var(X2, axis=0))[::-1]
    #X2 = X2[:, X2_var[:1000]]
    # Save pre-processed datasets
    #np.savetxt(path + 'TCGA_mRNAs_processed.csv', X1, delimiter=",")
    #np.savetxt(path + 'TCGA_miRNAs_processed.csv', X2, delimiter=",")

    # Load the 2 input datasets
    X1 = np.loadtxt(path + 'TCGA_mRNAs_processed.csv', delimiter=",")
    X2 = np.loadtxt(path + 'TCGA_miRNAs_processed.csv', delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    print(X1.shape, X2.shape)
    # Split into training and validation sets
    n_samples = X1.shape[0]
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:2500], indices[2500:]
    X1_train, X1_val = X1[train_idx,:], X1[val_idx,:]
    X2_train, X2_val = X2[train_idx,:], X2[val_idx,:]
    
    # Initialize Dataloader
    train_loader = data.DataLoader(
        ConcatDataset(X1_train, X2_train), 
        batch_size=64, shuffle=True, drop_last=False, num_workers=5)
    val_loader = data.DataLoader(
        ConcatDataset(X1_val, X2_val), 
        batch_size=64, shuffle=False, drop_last=False, num_workers=5)

    model = XVAE(X1.shape[1], X2.shape[1], ls=64, distance='kld', beta=1, save_model=False)
    # Initialize Trainer and setting parameters
    trainer = L.Trainer(accelerator="auto", devices=1, max_epochs=25, log_every_n_steps=10)
    # Use trainer to fit vae model to dataset
    trainer.fit(model, train_loader, val_loader)
    
    # Test best model on validation and test set
    #val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    #test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    
    
if __name__ == "__main__":
    main()
    
    
    
