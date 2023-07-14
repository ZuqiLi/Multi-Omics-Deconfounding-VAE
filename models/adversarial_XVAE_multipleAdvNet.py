import os
from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import ModuleList
import pytorch_lightning as L
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import pearsonr
from models.func import kld, mmd, reconAcc_pearsonCorr, reconAcc_relativeError, mse, crossEntropy, bce, init_weights
from models.clustering import *
import pandas as pd
import matplotlib.pyplot as plt


def seqBlock(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Dropout(*args, **kwargs),
        nn.PReLU(),
        nn.BatchNorm1d(out_f)
    )


class XVAE_adversarial_multinet(L.LightningModule):
    def __init__(self, 
                 PATH_xvae_ckpt,
                 PATH_advNet_ckpt,
                 dic_conf,
                 lamdba_deconf = 1,
                 distance="mmd", 
                 beta=1): 
        super().__init__()
        self.dic_conf = dic_conf
        self.lamdba_deconf = lamdba_deconf
        self.distance = distance
        self.beta = beta
        self.save_hyperparameters()
        self.test_step_outputs = [] 

        ### Load pre-trained XVAE model
        self.xvae = XVAE.load_from_checkpoint(PATH_xvae_ckpt)

        ### Load pre-trained advNet and freeze weights
        self.advNet_pre = advNet.load_from_checkpoint(PATH_advNet_ckpt)

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay=0)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_combined_loss"}

    def compute_loss_combined(self, batch):
        ''' 
        Autoencoder loss 
        '''
        recon_loss, reg_loss, _ = self.xvae.compute_loss(batch)
        ae_loss = recon_loss + reg_loss

        ''' 
        Adversarial net loss
        ''' 
        dic_adv_loss = self.advNet_pre.compute_loss(batch)
        advNet_loss = sum(dic_adv_loss.values())  

        '''
        Combined loss function for adversarial training
        (combined loss) = (autoencoder loss) - lambda * (adversarial loss)
        '''
        combined_loss = ae_loss - self.lamdba_deconf * advNet_loss

        return ae_loss, advNet_loss, combined_loss
    

    ### Ping pong training
    def training_step(self, batch, batch_idx):
        if self.current_epoch % 2 == 0:
            for param in self.xvae.parameters():
                param.requires_grad = True
            for param in self.advNet_pre.parameters():
                param.requires_grad = False        
            # Calculate loss
            ae_loss, advNet_loss, combined_loss = self.compute_loss_combined(batch)
            # Save metric 
            self.log('train_ae_loss', ae_loss, on_step = False, on_epoch = True, prog_bar = True)       
            self.log('train_combined_loss', combined_loss, on_step = False, on_epoch = True, prog_bar = True)
            # Prepare for next epoch
            return combined_loss
        else:             
            # Freeze that fucker
            for param in self.xvae.parameters():
                param.requires_grad = False
            for param in self.advNet_pre.parameters():
                param.requires_grad = True        
            advNet_loss = sum(self.advNet_pre.compute_loss(batch).values())
            # save metric
            self.log('train_advNet_loss', advNet_loss, on_step = False, on_epoch = True, prog_bar = True)
            return advNet_loss


    def validation_step(self, batch, batch_idx):
        ae_loss, advNet_loss, combined_loss = self.compute_loss_combined(batch)
        self.log('val_ae_loss', ae_loss, on_step = False, on_epoch = True, prog_bar = True)       
        self.log('val_advNet_loss', advNet_loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log('val_combined_loss', combined_loss, on_step = False, on_epoch = True, prog_bar = True)
        return combined_loss
    
    def test_step(self, batch, batch_idx):
        ''' Do a final quality check once using the external test set; Add here our other QC metrices; Relative Error, R2, clustering metric '''
        batch, y = batch[:3], batch[-1]
        ae_loss, advNet_loss, combined_loss = self.compute_loss_combined(batch)
        return combined_loss


class advNet(L.LightningModule):
    def __init__(self,
                 PATH_xvae_ckpt,
                 ls,
                 dic_conf,
                 loss_func_regr="mse"):
        super().__init__()
        self.ls = ls
        self.loss_func_regr = loss_func_regr
        self.dic_conf = dic_conf
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        self.save_hyperparameters()

        ### Load pre-trained XVAE model
        self.xvae = XVAE.load_from_checkpoint(PATH_xvae_ckpt)
        self.xvae.freeze()

        ### adversarial net
        self.adv_net_hidden = nn.Sequential(nn.Linear(self.ls, 10),
                                            nn.LeakyReLU())
        self.adv_net_allConfounders = []

        for key, val in self.dic_conf.items():
            if key.endswith("_CONT"):
                self.adv_net_allConfounders.append(nn.Sequential(nn.Linear(10, len(val)),
                                                                 nn.ReLU()))   
            if key.endswith("_OHE"):
                if len(val) > 1:  ## multilabel clf
                    self.adv_net_allConfounders.append(nn.Sequential(nn.Linear(10,  len(val)),
                                                       nn.Softmax()))     
                else: ## single label clf
                    self.adv_net_allConfounders.append(nn.Sequential(nn.Linear(10,  len(val)),
                                                       nn.Sigmoid()))            
                    
        ### This is needed for Lightning to realise that the subnets also get CUDA! 
        self.adv_net_allConfounders = ModuleList(self.adv_net_allConfounders)

        for ele in self.adv_net_allConfounders:
            ele.apply(lambda m: init_weights(m, "rai"))
        
        ### Initialise weights  
        for ele in [self.adv_net_hidden]:
            ele.apply(lambda m: init_weights(m, "rai"))
            
    def forward(self, x1, x2):
        z = self.xvae.generate_embedding(x1, x2)
        hidden = self.adv_net_hidden(z)
        y_pred_all = []
        for net in self.adv_net_allConfounders:
            y_pred_all.append(net(hidden))
        return y_pred_all

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay=0.001)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "advNet_val_total"}


    def compute_loss(self, batch):        
        x1, x2, allCov = batch
        y_pred_all = self.forward(x1,x2)

        total_loss = dict()
        for idx, [key, val] in enumerate(self.dic_conf.items()):
            cov = allCov[:,val].to(torch.float32)
            y_pred = y_pred_all[idx]

            ### Figure out correct loss function (depending on type of covariate)
            if key.endswith("_CONT"):
                total_loss[key] = mse(y_pred, cov.flatten()) 

            if key.endswith("_OHE"):
                if len(val) > 1:  ## multilabel clf
                    total_loss[key] = crossEntropy(y_pred, cov)
                else: ### single label
                    total_loss[key] =  bce(y_pred, cov) 
        return total_loss

    def training_step(self, batch, batch_idx):
        dict_loss = self.compute_loss(batch)
        loss = sum(dict_loss.values())
        self.log(f'advNet_train_total', loss, on_step = False, on_epoch = True, prog_bar = True)  
        return loss
    
    def validation_step(self, batch, batch_idx):
        dict_loss = self.compute_loss(batch)
        for key, val in dict_loss.items():
            self.log(f'advNet_val_{key}', val, on_step = False, on_epoch = True, prog_bar = True)   
        loss = sum(dict_loss.values())           
        self.log(f'advNet_val_total', loss, on_step = False, on_epoch = True, prog_bar = True)  
        return loss
    
    def test_step(self, batch, batch_idx):
        batch, y = batch[:3], batch[-1]
        y_pred_all = self.forward(batch[0],batch[1])        

        dict_loss = self.compute_loss(batch)
    
        loss = sum(dict_loss.values())   
        self.log('advNet_test_loss', loss, on_step = False, on_epoch = True)
        return loss


class XVAE(L.LightningModule):
    def __init__(self, 
                 input_size: list[int],
                 hidden_ind_size: list[int], 
                 hidden_fused_size: list[int],
                 ls: int, 
                 distance: str, 
                 beta=1,
                 lossReduction="mean",
                 klAnnealing=False,
                 dropout=0.2,
                 init_weights_func="rai") -> None:

        super().__init__()
        self.input_size = input_size
        self.hidden_ind_size = hidden_ind_size  
        self.hidden_fused_size = hidden_fused_size 
        self.ls = ls                    # latent size
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
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
            fused_encoder.append(nn.PReLU())
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
            decoder_layers.append(nn.PReLU())
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
        x1, x2, _ = batch
        mu, log_var = self.encode(x1, x2)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z)

        if self.distance == "mmd":
            true_samples = torch.randn([x1.shape[0], self.ls], device=z.device)
            distance = mmd(true_samples, z, reduction=self.lossReduction)        ### 'mean'
        if self.distance == "kld":
            distance = kld(mu, log_var, reduction=self.lossReduction)            ### 'mean'

        recon_loss_criterion = nn.MSELoss(reduction=self.lossReduction)  
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
        batch, y = batch[:3], batch[3]
        recon_loss, reg_loss, z = self.compute_loss(batch)
        loss = recon_loss + reg_loss
        x_hat = self.forward(*batch[:2])

        self.log('test_loss', loss , on_step = False, on_epoch = True)
        return loss




