import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
from models.func import init_weights, kld, mmd, reconAcc_pearsonCorr, reconAcc_relativeError
from models.clustering import *
import matplotlib.pyplot as plt


def seqBlock(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Dropout(*args, **kwargs),
        nn.PReLU(),
        nn.BatchNorm1d(out_f)
    )

class cXVAE_fusedEmbed(L.LightningModule):
    def __init__(self,
                 input_size: list[int],
                 hidden_ind_size: list[int], 
                 hidden_fused_size: list[int],
                 ls: int, 
                 cov_size: int, 
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
        self.ls = ls                                    # latent size
        self.cov_size = cov_size                        # number of covariates
        self.distance = distance                        # regularisation used
        self.beta = beta                                # weight for distance term in loss function
        self.num_clusters = 6                           # number of clusters  --> NOT GOOD IF HARDCODED!
        self.test_step_outputs = []                     # accumulate latent factors for all samples in every test step
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

        fused_encoder_all = [sum(self.hidden_ind_size) + self.cov_size] + self.hidden_fused_size            ### add covariates here
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

        decoder_topology = [self.ls + self.cov_size] + self.hidden_fused_size[::-1] + [sum(self.hidden_ind_size)]   ### add covariates in first layer here
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


    def encode(self, x1, x2, cov):
        cov = cov.reshape(-1, self.cov_size).to(torch.float32)
        x1_hidden = self.enc_hidden_x1(x1)
        x2_hidden = self.enc_hidden_x2(x2)
        x_fused = torch.cat((x1_hidden, x2_hidden, cov), dim=1)
        x_fused_hidden_2 = self.enc_hidden_fused(x_fused)
        mu = self.embed_mu(x_fused_hidden_2)
        log_var = self.embed_log_var(x_fused_hidden_2)  
        return mu, log_var
    
    def decode(self, z, cov):
        cov = cov.reshape(-1, self.cov_size).to(torch.float32)
        z_cov = torch.cat((z, cov),dim=1)
        x_fused_hat = self.decoder_fused(z_cov) 
        x1_hat = self.decoder_x1_hidden(x_fused_hat)
        x2_hat = self.decoder_x2_hidden(x_fused_hat)
        return x1_hat, x2_hat
    
    def forward(self, x1, x2, cov):
        mu, log_var = self.encode(x1, x2, cov)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z, cov)
        return x1_hat, x2_hat

    def generate_embedding(self, x1, x2, cov):
        mu, log_var = self.encode(x1, x2, cov)
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
        mu, log_var = self.encode(x1, x2, cov)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z, cov)

        if self.distance == "mmd":
            true_samples = torch.randn([x1.shape[0], self.ls], device=z.device)
            distance = mmd(true_samples, z, reduction='sum')
        if self.distance == "kld":
            distance = kld(mu, log_var, reduction='sum')         

        recon_loss_criterion = nn.MSELoss(reduction=self.lossReduction) 
        recon_loss_x1 = recon_loss_criterion(x1, x1_hat)
        recon_loss_x2 = recon_loss_criterion(x2, x2_hat)
        recon_loss = recon_loss_x1 + recon_loss_x2
        
        #vae_loss = recon_loss + self.beta * distance

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
        batch, y = batch[:3], batch[-1]
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

        # ''' Clustering performance '''
        # LF = torch.cat([x["z"] for x in self.test_step_outputs], 0)
        # Y = torch.cat([x["y"] for x in self.test_step_outputs], 0)
        # LF = LF.detach().cpu().numpy() # convert (GPU or CPU) tensor to numpy for the clustering
        # Y = Y.detach().cpu().numpy()
        # clust = kmeans(LF, self.num_clusters)
        # SS, DB = internal_metrics(LF, clust)
        # ARI, NMI = external_metrics(clust, Y)

        ''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
        x1 = torch.cat([x["x"][0] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x2 = torch.cat([x["x"][1] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        conf = torch.cat([x["x"][2] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
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
            | Reconstruction accuracy X1 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x1):.2f}+-{np.std(reconAcc_x1):.2f} |
            | Reconstruction accuracy X2 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x2):.2f}+-{np.std(reconAcc_x2):.2f} |
            | Reconstruction accuracy - Relative error (L2 norm)   | {relativeError:.2f} |                                    
        """
        table = '\n'.join(l.strip() for l in table.splitlines())
        self.logger.experiment.add_text("Results on test set", table,0)

        ''' Visualise embedding '''
        # self.logger.experiment.add_embedding(LF, metadata=Y)

        # ''' Clustering performance for the confounder '''
        # ARI_conf, NMI_conf = external_metrics(clust, conf.flatten().astype(int))

        # ''' Absolute correlation to confounding variables '''
        # labels = ['Confounder']
        # #labels = ['Gender', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Age', 'Race1', 'Race2', 'Race3']
        # #bins = conf[:, 1:]
        # #digits = np.argmax(bins, axis=1)
        # #conf = np.concatenate((conf[:,[0]], digits[:,None]), axis=1)
        
        # #bins = conf[:, 5:15]
        # #digits = np.argmax(bins, axis=1)
        # #conf = np.concatenate((conf[:,:5], digits[:,None], conf[:,15:]), axis=1)
        # corr_conf = [np.abs(np.corrcoef(LF.T, conf[:,i].T)[:-1,-1]) for i in range(conf.shape[1])]
        # fig, ax = plt.subplots(figsize=(15,5))
        # im = plt.imshow(corr_conf, cmap='hot', interpolation='nearest', vmin=0, vmax=0.5)
        # ax.set_yticks(np.arange(conf.shape[1]), labels=labels)
        # ax.tick_params(axis='both', labelsize=10)
        # plt.colorbar(im)
        # self.logger.experiment.add_figure(tag="Correlation with covariates", figure=fig)
        
        # ''' Association between clustering and confounders '''
        # pvals = test_confounding(clust, conf)
        
        # ''' Association between embeddings and confounders '''
        # fpvals, arsqs = test_embedding_confounding(LF, conf)

        # ''' Summary Table for tensorboard'''
        # table = f"""
        #     | Metric | Value  |
        #     |----------|-----------|
        #     | Silhouette score    | {SS:.2f} |
        #     | DB index    | {DB:.2f} |
        #     | Adjusted Rand Index   | {ARI:.2f} |
        #     | Normalized Mutual Info   | {NMI:.2f} |
        #     | Adjusted Rand Index (conf) | {ARI_conf:.2f} |
        #     | Normalized Mutual Info (conf) | {NMI_conf:.2f} |
        #     | Reconstruction accuracy X1 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x1):.2f}+-{np.std(reconAcc_x1):.2f} |
        #     | Reconstruction accuracy X2 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x2):.2f}+-{np.std(reconAcc_x2):.2f} |
        #     | Reconstruction accuracy - Relative error (L2 norm)   | {relativeError:.2f} |                                    
        # """
        # table = '\n'.join(l.strip() for l in table.splitlines())
        # for i in range(conf.shape[1]):
        #     table += f"| ANOVA between clustering and {labels[i]}  | {pvals[i]:.2e} |\n"
        # for i in range(conf.shape[1]):
        #     table += f"| F test between embedding and {labels[i]}  | {fpvals[i]:.2e} |\n"
        # for i in range(conf.shape[1]):
        #     table += f"| Adj. R-sq between embedding and {labels[i]}  | {arsqs[i]:.2e} |\n"
        # self.logger.experiment.add_text("Results on test set", table, 0)

        # ''' Visualise embedding '''        
        # ### conf_list = [conf[:,i] for i in range(conf.shape[1])]   ## There is an option to give multiple labels but I couldn't make it work
        # self.logger.experiment.add_embedding(LF, metadata=Y)

        return 





############################################################################################################################################################

# def init_weights(layer):
#     ''' Initialise layers for smoother training 
#         * Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a uniform distribution * '''
#     if isinstance(layer, nn.Linear):
#         torch.nn.init.kaiming_uniform_(layer.weight.data)


class cXVAE(L.LightningModule):
    def __init__(self, 
                 x1_size, 
                 x2_size, 
                 ls, 
                 cov_size, 
                 distance, 
                 beta): 
                 ### save_model):     ### NOTE: this will be taken over by Lightning

        super().__init__()
        self.ls = ls                    # latent size
        self.distance = distance        # regularisation used
        self.beta = beta                # weight for distance term in loss function
        self.cov_size = cov_size        # number of covariates
        self.num_clusters = 6           # number of clusters
        self.save_hyperparameters()     # save all hyperparameters to simplify model re-instantiation
        self.test_step_outputs = []     # accumulate latent factors for all samples in every test step
        
        ### encoder
        ### NOTE: hard coded reduction for now - change later!!
        #self.encoder_x1_fc = nn.Sequential(nn.Linear(x1_size + self.cov_size, 128), 
        self.encoder_x1_fc = nn.Sequential(nn.Linear(x1_size, 128), 
                                           nn.LeakyReLU(), 
                                           nn.BatchNorm1d(128))   
        #self.encoder_x2_fc = nn.Sequential(nn.Linear(x2_size + self.cov_size, 128), 
        self.encoder_x2_fc = nn.Sequential(nn.Linear(x2_size, 128), 
                                           nn.LeakyReLU(), 
                                           nn.BatchNorm1d(128))   
        ### fusing
        self.encoder_fuse = nn.Sequential(nn.Linear(128 + 128 + self.cov_size, 128), ### add covariates in this layer
        #self.encoder_fuse = nn.Sequential(nn.Linear(128 + 128, 128), 
                                          nn.LeakyReLU(), 
                                          nn.BatchNorm1d(128))  
        
        ### latent embedding
        self.embed_mu = nn.Linear(128, self.ls)
        self.embed_log_var = nn.Linear(128, self.ls)

        ### decoder
        self.decoder_sample = nn.Sequential(nn.Linear(self.ls + self.cov_size, 128),
        #self.decoder_sample = nn.Sequential(nn.Linear(self.ls, 128),
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


    def encode(self, x1, x2, cov):
        cov = cov.reshape(-1, self.cov_size).to(torch.float32)
        #x1 = self.encoder_x1_fc(torch.cat((x1, cov), dim=1))
        x1 = self.encoder_x1_fc(x1)
        #x2 = self.encoder_x2_fc(torch.cat((x2, cov), dim=1))
        x2 = self.encoder_x2_fc(x2)
        x_fused = torch.cat((x1, x2, cov), dim=1)
        #x_fused = torch.cat((x1, x2), dim=1)
        x_hidden = self.encoder_fuse(x_fused)
        mu = self.embed_mu(x_hidden)
        log_var = self.embed_log_var(x_hidden)
        return mu, log_var
    
    def decode(self, z, cov):
        cov = cov.reshape(-1, self.cov_size).to(torch.float32)
        z_cov = torch.cat((z, cov),dim=1)
        x_fused_hat = self.decoder_sample(z_cov)
        #x_fused_hat = self.decoder_sample(z)
        x1_hat = self.decoder_x1_fc(x_fused_hat)
        x2_hat = self.decoder_x2_fc(x_fused_hat)
        return x1_hat, x2_hat
    
    def forward(self, x1, x2, cov):
        mu, log_var = self.encode(x1, x2, cov)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z, cov)
        return x1_hat, x2_hat

    def generate_embedding(self, x1, x2, cov):
        mu, log_var = self.encode(x1, x2, cov)
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
        mu, log_var = self.encode(x1, x2, cov)
        z = self.sample_z(mu, log_var)
        x1_hat, x2_hat = self.decode(z, cov)

        if self.distance == "mmd":
            true_samples = torch.randn([x1.shape[0], self.ls], device=z.device)
            distance = mmd(true_samples, z)
        if self.distance == "kld":
            distance = kld(mu, log_var)         

        recon_loss_criterion = nn.MSELoss(reduction="sum")  ##### CHECK "mean" here again! "sum" better?
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
        batch, y = batch[:3], batch[-1]
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

        ''' Clustering performance '''
        LF = torch.cat([x["z"] for x in self.test_step_outputs], 0)
        Y = torch.cat([x["y"] for x in self.test_step_outputs], 0)
        LF = LF.detach().cpu().numpy() # convert (GPU or CPU) tensor to numpy for the clustering
        Y = Y.detach().cpu().numpy()
        clust = kmeans(LF, self.num_clusters)
        SS, DB = internal_metrics(LF, clust)
        ARI, NMI = external_metrics(clust, Y)

        ''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
        x1 = torch.cat([x["x"][0] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x2 = torch.cat([x["x"][1] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        conf = torch.cat([x["x"][2] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x1_hat = torch.cat([x["recon"][0] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        x2_hat = torch.cat([x["recon"][1] for x in self.test_step_outputs], 0).detach().cpu().numpy() 
        reconAcc_x1, reconAcc_x2 = reconAcc_pearsonCorr(x1, x1_hat, x2, x2_hat)

        ''' Relative Error using L2 norm '''
        relativeError = reconAcc_relativeError(x1, x1_hat,  x2, x2_hat)

        ''' Clustering performance for the confounder '''
        ARI_conf, NMI_conf = external_metrics(clust, conf.flatten().astype(int))

        ''' Absolute correlation to confounding variables '''
        labels = ['Confounder']
        #labels = ['Gender', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Age', 'Race1', 'Race2', 'Race3']
        #bins = conf[:, 1:]
        #digits = np.argmax(bins, axis=1)
        #conf = np.concatenate((conf[:,[0]], digits[:,None]), axis=1)
        
        #bins = conf[:, 5:15]
        #digits = np.argmax(bins, axis=1)
        #conf = np.concatenate((conf[:,:5], digits[:,None], conf[:,15:]), axis=1)
        corr_conf = [np.abs(np.corrcoef(LF.T, conf[:,i].T)[:-1,-1]) for i in range(conf.shape[1])]
        fig, ax = plt.subplots(figsize=(15,5))
        im = plt.imshow(corr_conf, cmap='hot', interpolation='nearest', vmin=0, vmax=0.5)
        ax.set_yticks(np.arange(conf.shape[1]), labels=labels)
        ax.tick_params(axis='both', labelsize=10)
        plt.colorbar(im)
        self.logger.experiment.add_figure(tag="Correlation with covariates", figure=fig)
        
        ''' Association between clustering and confounders '''
        pvals = test_confounding(clust, conf)
        
        ''' Association between embeddings and confounders '''
        fpvals, arsqs = test_embedding_confounding(LF, conf)

        ''' Summary Table for tensorboard'''
        table = f"""
            | Metric | Value  |
            |----------|-----------|
            | Silhouette score    | {SS:.2f} |
            | DB index    | {DB:.2f} |
            | Adjusted Rand Index   | {ARI:.2f} |
            | Normalized Mutual Info   | {NMI:.2f} |
            | Adjusted Rand Index (conf) | {ARI_conf:.2f} |
            | Normalized Mutual Info (conf) | {NMI_conf:.2f} |
            | Reconstruction accuracy X1 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x1):.2f}+-{np.std(reconAcc_x1):.2f} |
            | Reconstruction accuracy X2 - Pearson correlation (mean+-std)   | {np.mean(reconAcc_x2):.2f}+-{np.std(reconAcc_x2):.2f} |
            | Reconstruction accuracy - Relative error (L2 norm)   | {relativeError:.2f} |                                    
        """
        table = '\n'.join(l.strip() for l in table.splitlines())
        for i in range(conf.shape[1]):
            table += f"| ANOVA between clustering and {labels[i]}  | {pvals[i]:.2e} |\n"
        for i in range(conf.shape[1]):
            table += f"| F test between embedding and {labels[i]}  | {fpvals[i]:.2e} |\n"
        for i in range(conf.shape[1]):
            table += f"| Adj. R-sq between embedding and {labels[i]}  | {arsqs[i]:.2e} |\n"
        self.logger.experiment.add_text("Results on test set", table, 0)

        ''' Visualise embedding '''        
        ### conf_list = [conf[:,i] for i in range(conf.shape[1])]   ## There is an option to give multiple labels but I couldn't make it work
        self.logger.experiment.add_embedding(LF, metadata=Y)

        return 


