import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
from models.func import kld, mmd, reconAcc_pearsonCorr, reconAcc_relativeError
from models.clustering import kmeans, internal_metrics, external_metrics


def init_weights(layer):
    ''' Initialise layers for smoother training 
        * Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a uniform distribution * '''
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight.data)


class XVAE(L.LightningModule):
    def __init__(self, 
                 x1_size, 
                 x2_size, 
                 ls, 
                 distance, 
                 beta): 
                 ### save_model):     ### NOTE: this will be taken over by Lightning

        super().__init__()
        self.ls = ls                    # latent size
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
        self.c = 6                      # number of clusters
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        
        ### encoder
        ### NOTE: hard coded reduction for now - change later!!
        self.encoder_x1_fc = nn.Sequential(nn.Linear(x1_size, 128), 
                                           nn.LeakyReLU(), 
                                           nn.BatchNorm1d(128))   
        self.encoder_x2_fc = nn.Sequential(nn.Linear(x2_size, 128), 
                                           nn.LeakyReLU(), 
                                           nn.BatchNorm1d(128))   
        ### fusing
        self.encoder_fuse = nn.Sequential(nn.Linear(128+128,     
                                                    128), 
                                          nn.LeakyReLU(), 
                                          nn.BatchNorm1d(128))  
        
        ### latent embedding
        self.embed_mu = nn.Linear(128, self.ls)
        self.embed_log_var = nn.Linear(128, self.ls)

        ### decoder
        self.decoder_sample = nn.Sequential(nn.Linear(self.ls, 128),
                                            nn.LeakyReLU())
        self.decoder_x1_fc = nn.Sequential(nn.Linear(128, x1_size),
                                           nn.Sigmoid())
        self.decoder_x2_fc = nn.Sequential(nn.Linear(128, x2_size),
                                           nn.Sigmoid())

        ### Initialise weights
        for ele in [self.encoder_x1_fc, self.encoder_x2_fc, self.encoder_fuse, self.embed_mu, self.embed_log_var, self.decoder_sample, self.decoder_x1_fc, self.decoder_x2_fc]:
            ele.apply(init_weights)


    def sample_z(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z


    def encode(self, x1, x2):
        x1 = self.encoder_x1_fc(x1)
        x2 = self.encoder_x2_fc(x2)
        x_fused = torch.cat((x1, x2), dim=1)
        x_hidden = self.encoder_fuse(x_fused)
        mu = self.embed_mu(x_hidden)
        log_var = self.embed_log_var(x_hidden)
        return mu, log_var


    def decode(self, z):
        x_fused_hat = self.decoder_sample(z)
        x1_hat = self.decoder_x1_fc(x_fused_hat)
        x2_hat = self.decoder_x2_fc(x_fused_hat)
        return x1_hat, x2_hat


    def forward(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z)
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
            true_samples = torch.randn([x1.shape[0], self.ls], device=z.device)
            distance = mmd(true_samples, z)
        if self.distance == "kld":
            distance = kld(mu, log_var)         

        recon_loss_criterion = nn.MSELoss(reduction="mean")  ##### CHECK "mean" here again! "sum" better?
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss_x2 = recon_loss_criterion(x2, x2_hat)
        recon_loss = recon_loss_x1 + recon_loss_x2
        
        #vae_loss = recon_loss + self.beta * distance

        ### Implement (very easy, monotonic) KL annealing - slowly start increasing beta value
        if self.current_epoch <= 10:
            self.beta = 0
        elif (self.current_epoch > 10) & (self.current_epoch < 20):   #### possibly change these values
            self.beta = 0.5
        else:
            self.beta = 1

        reg_loss = self.beta * distance

        return recon_loss, reg_loss, z


    def training_step(self, batch, batch_idx):
        recon_loss, reg_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss
        self.log('train_recon_loss', recon_loss, on_step = False, on_epoch = True, prog_bar = True)       
        self.log('train_reg_loss', reg_loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss


    def validation_step(self, batch, batch_idx):
        recon_loss, reg_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss
        self.log('val_loss', loss, on_step = False, on_epoch = True)
        return loss


    def test_step(self, batch, batch_idx):
        ''' Do a final quality check once using the external test set; Add here our other QC metrices; Relative Error, R2, clustering metric '''
        batch, y = batch[:2], batch[2]
        recon_loss, reg_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss
        x_hat = self.forward(*batch)

        self.log('test_loss', loss , on_step = False, on_epoch = True)
        self.test_step_outputs.append({"z": z, "y": y, "recon": x_hat, "x": batch})
        return loss
    

    def on_test_epoch_end(self):
        '''
        Quality checks on Test set: 
            - Clustering:
                - ... 
        '''
        print("\n\n ON_TEST_EPOCH_END\n\n")

        ''' Clustering '''
        LF = torch.cat([x["z"] for x in self.test_step_outputs], 0)
        Y = torch.cat([x["y"] for x in self.test_step_outputs], 0)
        LF = LF.detach().cpu().numpy() # convert (GPU or CPU) tensor to numpy for the clustering
        Y = Y.detach().cpu().numpy()
        clust = kmeans(LF, self.c)
        SS, DB = internal_metrics(LF, clust)
        ARI, NMI = external_metrics(clust, Y)

        ''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
        x1 = torch.cat([x["x"][0] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x2 = torch.cat([x["x"][1] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x1_hat = torch.cat([x["recon"][0] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x2_hat = torch.cat([x["recon"][1] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        reconAcc_x1, reconAcc_x2 = reconAcc_pearsonCorr(x1, x1_hat, x2, x2_hat)

        ''' Relative Error using L2 norm '''
        relativeError = reconAcc_relativeError(x1, x1_hat,  x2, x2_hat)


        ''' Summary Table for tensorboard'''
        table = f"""
            | Metric | Value  |
            |----------|-----------|
            | Silhouette score    | {SS:.2f} |
            | DB index    | {DB:.2f} |
            | Adjusted Rand Index   | {ARI:.2f} |
            | Normalized Mutual Info   | {NMI:.2f} |
            | Reconstruction accuracy X1 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x1):.2f}+-{np.std(reconAcc_x1):.2f} |
            | Reconstruction accuracy X2 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x2):.2f}+-{np.std(reconAcc_x2):.2f} |
            | Reconstruction accuracy - Relative error (L2 norm)   | {relativeError:.2f} |                                    
        """
        table = '\n'.join(l.strip() for l in table.splitlines())
        self.logger.experiment.add_text("Results on test set", table,0)

        ''' Visualise embedding '''
        self.logger.experiment.add_embedding(LF, metadata=Y)

        return 





class VAE(L.LightningModule):
    def __init__(self, 
                 x1_size, 
                 x2_size, 
                 ls, 
                 distance, 
                 beta): 
                 ### save_model):     ### NOTE: this will be taken over by Lightning

        super().__init__()
        self.ls = ls                    # latent size
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
        self.c = 6                      # number of clusters
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        
        ### encoder
        ### NOTE: hard coded reduction for now - change later!!
        self.encoder_x1_fc = nn.Sequential(nn.Linear(x1_size, 128), 
                                           nn.LeakyReLU(), 
                                           nn.BatchNorm1d(128))   
        
        ### latent embedding
        self.embed_mu = nn.Linear(128, self.ls)
        self.embed_log_var = nn.Linear(128, self.ls)

        ### decoder
        self.decoder_sample = nn.Sequential(nn.Linear(self.ls, 128),
                                            nn.LeakyReLU())
        self.decoder_x1_fc = nn.Sequential(nn.Linear(128, x1_size),
                                           nn.Sigmoid())

        ### Initialise weights
        for ele in [self.encoder_x1_fc, self.embed_mu, self.embed_log_var, self.decoder_sample, self.decoder_x1_fc]:
            ele.apply(init_weights)


    def sample_z(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z


    def encode(self, x1, x2):
        x_hidden = self.encoder_x1_fc(x1)
        mu = self.embed_mu(x_hidden)
        log_var = self.embed_log_var(x_hidden)
        return mu, log_var


    def decode(self, z):
        x_fused_hat = self.decoder_sample(z)
        x1_hat = self.decoder_x1_fc(x_fused_hat)
        return x1_hat


    def forward(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat = self.decode(z)
        return x1_hat


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
        x1_hat = self.decode(z)

        if self.distance == "mmd":
            true_samples = torch.randn([x1.shape[0], self.ls], device=z.device)
            distance = mmd(true_samples, z)
        if self.distance == "kld":
            distance = kld(mu, log_var)         
    
        recon_loss_criterion = nn.MSELoss(reduction="mean")  ##### CHECK "mean" here again! "sum" better?
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss = recon_loss_x1
        
        vae_loss = recon_loss + self.beta * distance
        return vae_loss, z


    def training_step(self, batch, batch_idx):
        loss, z = self.compute_loss(batch)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, z = self.compute_loss(batch)
        self.log('val_loss', loss , on_step = False, on_epoch = True)
        return loss


    def test_step(self, batch, batch_idx):
        ''' Do a final quality check once using the external test set; Add here our other QC metrices; Relative Error, R2, clustering metric '''
        batch, y = batch[:2], batch[2]
        loss, z = self.compute_loss(batch)
        self.log('test_loss', loss , on_step = False, on_epoch = True)
        self.test_step_outputs.append({"z": z, "y": y})
        return loss
    

    def on_test_epoch_end(self):
        LF = torch.cat([x["z"] for x in self.test_step_outputs], 0)
        Y = torch.cat([x["y"] for x in self.test_step_outputs], 0)
        LF = LF.detach().cpu().numpy() # convert (GPU or CPU) tensor to numpy for the clustering
        Y = Y.detach().cpu().numpy()
        clust = kmeans(LF, self.c)
        SS, DB = internal_metrics(LF, clust)
        ARI, NMI = external_metrics(clust, Y)
        self.log("Silhouette score", SS, on_step = False, on_epoch = True)
        self.log("DB index", DB, on_step = False, on_epoch = True)
        self.log("Adjusted Rand Index", ARI, on_step = False, on_epoch = True)
        self.log("Normalized Mutual Info", NMI, on_step = False, on_epoch = True)
        return {"Silhouette score": SS, "DB index": DB, "Adjusted Rand Index": ARI, "Normalized Mutual Info": NMI}


class AE(L.LightningModule):
    def __init__(self, 
                 x1_size, 
                 x2_size, 
                 ls, 
                 distance, 
                 beta): 
                 ### save_model):     ### NOTE: this will be taken over by Lightning

        super().__init__()
        self.ls = ls                    # latent size
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
        self.c = 6                      # number of clusters
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        
        ### encoder
        ### NOTE: hard coded reduction for now - change later!!
        self.encoder_x1_fc = nn.Sequential(nn.Linear(x1_size, 128), 
                                           nn.LeakyReLU(), 
                                           nn.BatchNorm1d(128))   
        
        ### latent embedding
        self.embed = nn.Sequential(nn.Linear(128, self.ls),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(self.ls))

        ### decoder
        self.decoder_hidden = nn.Sequential(nn.Linear(self.ls, 128),
                                            nn.LeakyReLU())
        self.decoder_x1_fc = nn.Sequential(nn.Linear(128, x1_size),
                                           nn.Sigmoid())

        ### Initialise weights
        for ele in [self.encoder_x1_fc, self.embed, self.decoder_hidden, self.decoder_x1_fc]:
            ele.apply(init_weights)


    def encode(self, x1, x2):
        x_hidden = self.encoder_x1_fc(x1)
        z = self.embed(x_hidden)
        return z


    def decode(self, z):
        x_fused_hat = self.decoder_hidden(z)
        x1_hat = self.decoder_x1_fc(x_fused_hat)
        return x1_hat


    def forward(self, x1, x2):
        z = self.encode(x1, x2)
        x1_hat = self.decode(z)
        return x1_hat


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay=0.001)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def compute_loss(self, batch):
        x1, x2 = batch
        z = self.encode(x1, x2)
        x1_hat = self.decode(z)

        recon_loss_criterion = nn.MSELoss(reduction="mean")  ##### CHECK "mean" here again! "sum" better?
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss = recon_loss_x1
        
        ae_loss = recon_loss
        return ae_loss, z


    def training_step(self, batch, batch_idx):
        loss, z = self.compute_loss(batch)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, z = self.compute_loss(batch)
        self.log('val_loss', loss , on_step = False, on_epoch = True)
        return loss


    def test_step(self, batch, batch_idx):
        ''' Do a final quality check once using the external test set; Add here our other QC metrices; Relative Error, R2, clustering metric '''
        batch, y = batch[:2], batch[2]
        loss, z = self.compute_loss(batch)
        self.log('test_loss', loss , on_step = False, on_epoch = True)
        self.test_step_outputs.append({"z": z, "y": y})
        return loss
    

    def on_test_epoch_end(self):
        LF = torch.cat([x["z"] for x in self.test_step_outputs], 0)
        Y = torch.cat([x["y"] for x in self.test_step_outputs], 0)
        LF = LF.detach().cpu().numpy() # convert (GPU or CPU) tensor to numpy for the clustering
        Y = Y.detach().cpu().numpy()
        clust = kmeans(LF, self.c)
        SS, DB = internal_metrics(LF, clust)
        ARI, NMI = external_metrics(clust, Y)
        self.log("Silhouette score", SS, on_step = False, on_epoch = True)
        self.log("DB index", DB, on_step = False, on_epoch = True)
        self.log("Adjusted Rand Index", ARI, on_step = False, on_epoch = True)
        self.log("Normalized Mutual Info", NMI, on_step = False, on_epoch = True)
        return {"Silhouette score": SS, "DB index": DB, "Adjusted Rand Index": ARI, "Normalized Mutual Info": NMI}


