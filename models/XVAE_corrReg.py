import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
from models.func import init_weights, kld, mmd, correlation, mutualInfo, reconAcc_pearsonCorr, reconAcc_relativeError
from models.clustering import kmeans, internal_metrics, external_metrics


def seqBlock(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Dropout(*args, **kwargs),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_f)
    )

class XVAE_corrReg(L.LightningModule):
    def __init__(self, 
                 input_size: list[int],
                 hidden_ind_size: list[int], 
                 hidden_fused_size: list[int],
                 ls: int, 
                 cov_size: int,
                 distance: str, 
                 beta=1,
                 lossReduction="sum",
                 klAnnealing=False,
                 dropout=0.2,
                 init_weights_func="rai") -> None:

        super().__init__()
        self.input_size = input_size
        self.hidden_ind_size = hidden_ind_size  
        self.hidden_fused_size = hidden_fused_size 
        self.ls = ls                    # latent size
        self.cov_size = cov_size        # number of covariates
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
        self.c = 6                      # number of clusters
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        self.lossReduction = lossReduction
        self.klAnnealing = klAnnealing
        self.dropout = dropout
        self.init_weights_func = init_weights_func
        self.save_hyperparameters()

        #################################################
        ##                   Encoder                   ##
        #################################################

        ### Individual omics part in encoder
        self.enc_hidden_x1 = seqBlock(self.input_size[0], self.hidden_ind_size[0], p=self.dropout)
        self.enc_hidden_x2 = seqBlock(self.input_size[1], self.hidden_ind_size[1], p=self.dropout)

        ### Fused layer reductions
        fused_encoder_all = [sum(self.hidden_ind_size)] + self.hidden_fused_size
        fused_encoder = []
        for i in range(len(fused_encoder_all)-1):
            layer = nn.Linear(fused_encoder_all[i], fused_encoder_all[i+1])
            layer.apply(lambda m: init_weights(m, self.init_weights_func))    
            fused_encoder.append(layer)
            fused_encoder.append(nn.Dropout(p=self.dropout))
            fused_encoder.append(nn.LeakyReLU())
            fused_encoder.append(nn.BatchNorm1d(fused_encoder_all[i+1]))      
        self.enc_hidden_fused = nn.Sequential(*fused_encoder)

        self.embed_mu = nn.Sequential(nn.Linear(self.hidden_fused_size[-1], self.ls))
        self.embed_log_var = nn.Sequential(nn.Linear(self.hidden_fused_size[-1], self.ls))

        #################################################
        ##                   Decoder                   ##
        #################################################

        decoder_topology = [self.ls] + self.hidden_fused_size[::-1] + [sum(self.hidden_ind_size)]
        decoder_layers = []
        for i in range(len(decoder_topology)-1):
            layer = nn.Linear(decoder_topology[i],decoder_topology[i+1])
            layer.apply(lambda m: init_weights(m, self.init_weights_func))    
            decoder_layers.append(layer)
            decoder_layers.append(nn.Dropout(p=self.dropout))
            decoder_layers.append(nn.LeakyReLU())
        self.decoder_fused = nn.Sequential(*decoder_layers)

        self.decoder_x1_hidden = nn.Sequential(nn.Linear(decoder_topology[-1], self.input_size[0]),
                                            nn.Sigmoid())
        self.decoder_x2_hidden = nn.Sequential(nn.Linear(decoder_topology[-1], self.input_size[1]),
                                            nn.Sigmoid())


        ### Initialise weights
        for ele in [self.enc_hidden_x1, self.enc_hidden_x2, self.enc_hidden_fused, self.embed_mu, self.embed_log_var, 
                    self.decoder_fused, self.decoder_x1_hidden, self.decoder_x2_hidden]:
            ele.apply(init_weights)


    def sample_z(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z

    def encode(self, x1, x2):
        x1_hidden = self.enc_hidden_x1(x1)
        x2_hidden = self.enc_hidden_x2(x2)
        x_fused = torch.cat((x1_hidden, x2_hidden), dim=1)
        x_fused_hidden_2 = self.enc_hidden_fused(x_fused)
        mu = self.embed_mu(x_fused_hidden_2)
        log_var = self.embed_log_var(x_fused_hidden_2)   
        return mu, log_var


    def decode(self, z):
        x_fused_hat = self.decoder_fused(z) 
        x1_hat = self.decoder_x1_hidden(x_fused_hat)
        x2_hat = self.decoder_x2_hidden(x_fused_hat)
        return x1_hat, x2_hat


    def forward(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z)
        return x1_hat, x2_hat        
    

    def generate_embedding(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        return z


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay=0.001)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def compute_loss(self, batch):
        x1, x2, cov = batch
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z)

        if self.distance == "mmd":
            true_samples = torch.randn([x1.shape[0], self.ls], device=z.device)
            distance = mmd(true_samples, z, reduction='sum')        ### 'mean'
        if self.distance == "kld":
            distance = kld(mu, log_var, reduction='sum')            ### 'mean'

        recon_loss_criterion = nn.MSELoss(reduction=self.lossReduction)  ##### CHECK "mean" here again! "sum" better?
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss_x2 = recon_loss_criterion(x2, x2_hat)
        recon_loss = recon_loss_x1 + recon_loss_x2

        if self.klAnnealing:
            ### Implement (very easy, monotonic) KL annealing - slowly start increasing beta value
            if self.current_epoch <= 10:
                self.beta = 0
            elif (self.current_epoch > 10) & (self.current_epoch < 20):   #### possibly change these values
                self.beta = 0.5
            else:
                self.beta = 1
        reg_loss = self.beta * distance

        # Regularization for correlation between embedding and confounder
        #corr_loss = correlation(z, cov, nonneg='square')
        corr_loss = mutualInfo(z, cov)

        return recon_loss, reg_loss, corr_loss, z


    def training_step(self, batch, batch_idx):
        recon_loss, reg_loss, corr_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss + corr_loss
        self.log('train_recon_loss', recon_loss, on_step = False, on_epoch = True, prog_bar = True)       
        self.log('train_reg_loss', reg_loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log('train_corr_loss', corr_loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss


    def validation_step(self, batch, batch_idx):
        recon_loss, reg_loss, corr_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss + corr_loss
        self.log('val_loss', loss, on_step = False, on_epoch = True)
        return loss


    def test_step(self, batch, batch_idx):
        ''' Do a final quality check once using the external test set; Add here our other QC metrices; Relative Error, R2, clustering metric '''
        batch, y = batch[:3], batch[3]
        recon_loss, reg_loss, corr_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss + corr_loss
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
        
        reconAcc_x1 = reconAcc_pearsonCorr(x1, x1_hat)
        reconAcc_x2 = reconAcc_pearsonCorr(x2, x2_hat)

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



class XVAE_sym(L.LightningModule):
    def __init__(self, 
                 input_size: list[int],
                 hidden_ind_size: list[int], 
                 hidden_fused_size: list[int],
                 ls: int, 
                 distance: str, 
                 beta=1,
                 lossReduction="mean",
                 init_weights_func="rai",
                 klAnnealing=False) -> None:

        super().__init__()
        self.input_size = input_size
        self.hidden_ind_size = hidden_ind_size  
        self.hidden_fused_size = hidden_fused_size 
        self.ls = ls                    # latent size
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
        self.c = 6                      # number of clusters
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        self.lossReduction = lossReduction
        self.klAnnealing = klAnnealing
        self.init_weights_func = init_weights_func
        self.save_hyperparameters()

        #################################################
        ##                   Encoder                   ##
        #################################################

        ### Individual omics part in encoder
        self.enc_hidden_x1 = seqBlock(self.input_size[0], self.hidden_ind_size[0], p=0.2)
        self.enc_hidden_x2 = seqBlock(self.input_size[1], self.hidden_ind_size[1], p=0.2)

        ### Fused layer reductions
        fused_encoder_all = [sum(self.hidden_ind_size)] + self.hidden_fused_size
        fused_encoder = []
        for i in range(len(fused_encoder_all)-1):
            layer = nn.Linear(fused_encoder_all[i], fused_encoder_all[i+1])
            layer.apply(lambda m: init_weights(m, self.init_weights_func))    
            fused_encoder.append(layer)
            fused_encoder.append(nn.Dropout(p=0.2))
            fused_encoder.append(nn.PReLU())
            fused_encoder.append(nn.BatchNorm1d(fused_encoder_all[i+1]))      
        self.enc_hidden_fused = nn.Sequential(*fused_encoder)

        self.embed_mu = nn.Sequential(nn.Linear(self.hidden_fused_size[-1], self.ls))
        self.embed_log_var = nn.Sequential(nn.Linear(self.hidden_fused_size[-1], self.ls))

        #################################################
        ##                   Decoder                   ##
        #################################################

        decoder_topology = [self.ls] + self.hidden_fused_size[::-1] #+ [sum(self.hidden_ind_size)]
        decoder_layers = []
        for i in range(len(decoder_topology)-1):
            layer = nn.Linear(decoder_topology[i],decoder_topology[i+1])
            layer.apply(lambda m: init_weights(m, self.init_weights_func))    
            decoder_layers.append(layer)
            decoder_layers.append(nn.Dropout(p=0.2))
            decoder_layers.append(nn.PReLU())
        self.decoder_fused = nn.Sequential(*decoder_layers)

        self.decoder_x1_hidden = nn.Sequential(nn.Linear(decoder_topology[-1], self.hidden_ind_size[0]),
                                              nn.Dropout(p=0.2),
                                              nn.PReLU(),
                                              nn.Linear(self.hidden_ind_size[0], self.input_size[0]),
                                              nn.Sigmoid())
        self.decoder_x2_hidden = nn.Sequential(nn.Linear(decoder_topology[-1], self.hidden_ind_size[1]),
                                              nn.Dropout(p=0.2),
                                              nn.PReLU(),
                                              nn.Linear(self.hidden_ind_size[1], self.input_size[1]),
                                              nn.Sigmoid())


        ### Initialise weights
        for ele in [self.enc_hidden_x1, self.enc_hidden_x2, self.enc_hidden_fused, self.embed_mu, self.embed_log_var, 
                    self.decoder_fused, self.decoder_x1_hidden, self.decoder_x2_hidden]:
            ele.apply(init_weights)


    def sample_z(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z

    def encode(self, x1, x2):
        x1_hidden = self.enc_hidden_x1(x1)
        x2_hidden = self.enc_hidden_x2(x2)
        x_fused = torch.cat((x1_hidden, x2_hidden), dim=1)
        x_fused_hidden_2 = self.enc_hidden_fused(x_fused)
        mu = self.embed_mu(x_fused_hidden_2)
        log_var = self.embed_log_var(x_fused_hidden_2)   
        return mu, log_var


    def decode(self, z):
        x_fused_hat = self.decoder_fused(z) 
        x1_hat = self.decoder_x1_hidden(x_fused_hat)
        x2_hat = self.decoder_x2_hidden(x_fused_hat)
        return x1_hat, x2_hat


    def forward(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z)
        return x1_hat, x2_hat        
    

    def generate_embedding(self, x1, x2):
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        return z


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
            distance = mmd(true_samples, z, reduction='sum')        ### 'mean'
        if self.distance == "kld":
            distance = kld(mu, log_var, reduction='sum')            ### 'mean'

        recon_loss_criterion = nn.MSELoss(reduction=self.lossReduction)  ##### CHECK "mean" here again! "sum" better?
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss_x2 = recon_loss_criterion(x2, x2_hat)
        recon_loss = recon_loss_x1 + recon_loss_x2

        if self.klAnnealing:
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
        
        reconAcc_x1 = reconAcc_pearsonCorr(x1, x1_hat)
        reconAcc_x2 = reconAcc_pearsonCorr(x2, x2_hat)

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



