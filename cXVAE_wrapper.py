import os
import numpy as np
import pandas as pd
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
sys.path.append("./")
from models.cXVAE import cXVAE_fusedEmbed # cXVAE_input, cXVAE_inputEmbed, cXVAE_embed, cXVAE_fusedEmbed
from models.clustering import *
from Data.preprocess import *
from models.func import reconAcc_relativeError
import argparse


modelname = 'cXVAE'
maxEpochs = 20

#################################################
##             Training procedure              ##
#################################################
def training(train_loader, val_loader, X1_dim, X2_dim, conf_dim):
    # Initialize model
    model = cXVAE_fusedEmbed(input_size = [X1_dim, X2_dim],
                # first hidden layer: individual encoding of X1 and X2; [layersizeX1, layersizeX2]; length: number of input modalities
                hidden_ind_size =[200, 200],
                # next hidden layer(s): densely connected layers of fused X1 & X2; [layer1, layer2, ...]; length: number of hidden layers
                hidden_fused_size = [200],
                ls=50,                      # latent size
                cov_size=conf_dim,    # number of covariates
                distance='mmd',             # variational term: KL divergence or maximum mean discrepancy
                lossReduction='sum',        # sum or mean reduction for loss terms
                klAnnealing=False,          # annealing for KL vanishing
                beta=1,                     # weight for variational term
                dropout=0.2,                # dropout rate
                init_weights_func="rai")    # weight initialization
    print(model)
    # Initialize Trainer
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{modelname}/")
    trainer = L.Trainer(default_root_dir=os.getcwd(), 
                        accelerator="auto", 
                        devices=1, 
                        log_every_n_steps=10, 
                        logger=logger, 
                        max_epochs=maxEpochs,
                        fast_dev_run=False,
                        deterministic=True)
    trainer.fit(model, train_loader, val_loader)


###############################################
##         Test on the whole dataset         ##
###############################################
def testing(X1, X2, Y, conf):
    X1_test = scale(X1)
    X2_test = scale(X2)
    conf_test = conf
    conf = conf.detach().numpy()
    labels = ['Confounder'+str(i) for i in range(n_confs)]

    RE_X1s, RE_X2s, RE_X1X2s = [], [], []
    clusts = []
    SSs, DBs = [], []
    n_clust = len(np.unique(Y))
    corr_diff = []
    # Sample multiple times from the latent distribution for stability
    for i in range(150):
        ckpt_path = f"{os.getcwd()}/lightning_logs/{modelname}/version_0/checkpoints"
        ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"

        model = cXVAE_fusedEmbed.load_from_checkpoint(ckpt_file, map_location=torch.device('cpu'))
        
        # Loop over dataset and test on batches
        indices = np.array_split(np.arange(X1_test.shape[0]), 20)
        z = []
        X1_hat, X2_hat = [], []
        for idx in indices:
            z_batch = model.generate_embedding(X1_test[idx], X2_test[idx], conf_test[idx])
            z.append(z_batch.detach().numpy())
            X1_hat_batch, X2_hat_batch = model.decode(z_batch, conf_test[idx])
            X1_hat.append(X1_hat_batch.detach().numpy())
            X2_hat.append(X2_hat_batch.detach().numpy())

        z = np.concatenate(z)
        X1_hat = np.concatenate(X1_hat)
        X2_hat = np.concatenate(X2_hat)

        # Compute relative error from the last epoch
        RE_X1, RE_X2, RE_X1X2 = reconAcc_relativeError(X1_test, X1_hat, X2_test, X2_hat)
        RE_X1s.append(RE_X1)
        RE_X2s.append(RE_X2)
        RE_X1X2s.append(RE_X1X2)
        # Clustering the latent vectors from the last epoch
        clust = kmeans(z, n_clust)
        clusts.append(clust)
        # Compute clustering metrics
        SS, DB = internal_metrics(z, clust)
        SSs.append(SS)
        DBs.append(DB)

    # Average relative errors over all samplings
    print("Relative error (X1):", np.mean(RE_X1s))
    print("Relative error (X2):", np.mean(RE_X2s))
    print("Relative error (X1X2):", np.mean(RE_X1X2s))

    # Average clustering metrics over all samplings
    print("Silhouette score:", np.mean(SSs))
    print("DB index:", np.mean(DBs))
    # Compute consensus clustering from all samplings
    con_clust, _, disp = consensus_clustering(clusts, n_clust)
    print("Dispersion for co-occurrence matrix:", disp)
    # Compute ARI and NMI for cancer types
    ARI, NMI = external_metrics(con_clust, Y)
    print("ARI for cancer types:", ARI)
    print("NMI for cancer types:", NMI)
    # Compute ARI and NMI for each confounder
    if conf_type == 'categ':
        conf = np.argmax(conf, 1)[:,None]
    elif conf_type == 'multi':
        conf_categ = np.argmax(conf[:,(n_confs-1):], 1)
        conf = np.concatenate((conf[:,:(n_confs-1)], conf_categ[:,None]), axis=1)
    ARI_conf, NMI_conf = [], []
    for c in conf.T:
        ARI_c, NMI_c = external_metrics(con_clust, c)
        ARI_conf.append(ARI_c)
        NMI_conf.append(NMI_c)
    print("ARI for confounder:", ARI_conf)
    print("NMI for confounder:", NMI_conf)

    ### Save
    res = {'RelErr_X1':[np.mean(RE_X1s)],
        'RelErr_X2':[np.mean(RE_X2s)],
        'RelErr_X1X2':[np.mean(RE_X1X2s)],
        'CC_dispersion':[disp],
        'ss':[np.mean(SSs)],
        'db':[np.mean(DBs)],
        'ari_trueCluster':[ARI],
        'nmi_trueCluster':[NMI],
        'ari_confoundedCluster':[ARI_conf],
        'nmi_confoundedCluster':[NMI_conf]
        }

    pd.DataFrame(res).to_csv(f"lightning_logs/{modelname}/version_0/results_performance.csv", index=False)



if __name__ == '__main__':
    ''' Set seeds for replicability  -Ensure that all operations are deterministic on GPU (if used) for reproducibility '''
    np.random.seed(1234)
    torch.manual_seed(1234)
    L.seed_everything(1234, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ''' Load in data '''
    parser = argparse.ArgumentParser()
    parser.add_argument('pathname', type=str)
    parser.add_argument('X1_filename', type=str)
    parser.add_argument('X2_filename', type=str)
    parser.add_argument('Y_filename', type=str)
    parser.add_argument('conf_type', type=str)
    parser.add_argument('conf_filename', type=str)
    args = parser.parse_args()

    X1 = np.loadtxt(os.path.join(args.pathname, args.X1_filename), delimiter=",")
    X2 = np.loadtxt(os.path.join(args.pathname, args.X2_filename), delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    Y = np.loadtxt(os.path.join(args.pathname, args.Y_filename), delimiter=",")

    # load artificial confounder
    conf_type = args.conf_type
    conf = np.loadtxt(os.path.join(args.pathname, args.conf_filename), delimiter=",")
    conf = torch.from_numpy(conf).to(torch.float32)
    n_confs = conf.shape[1] if conf.ndim > 1 else 1
    if conf_type == 'conti':
        conf = conf[:,None]
    elif conf_type == 'categ':
        conf = torch.nn.functional.one_hot(conf.to(torch.int64))
    elif conf_type == 'multi':
        conf_categ = torch.nn.functional.one_hot(conf[:,-1].to(torch.int64))
        conf = torch.cat((conf[:,:-1], conf_categ), dim=1)
    print('Shape of confounders:', conf.shape)

    ''' Split into training and validation sets '''
    n_samples = X1.shape[0]
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:2000], indices[2000:]

    X1_train, X1_val = scale(X1[train_idx,:]), scale(X1[val_idx,:])
    X2_train, X2_val = scale(X2[train_idx,:]), scale(X2[val_idx,:])
    conf_train, conf_val = conf[train_idx,:], conf[val_idx,:]

    ''' Initialize Dataloader '''
    train_loader = data.DataLoader(
                    ConcatDataset(X1_train, X2_train, conf_train), 
                    batch_size=64, 
                    shuffle=True, 
                    drop_last=False, 
                    num_workers=5)
    val_loader = data.DataLoader(
                    ConcatDataset(X1_val, X2_val, conf_val), 
                    batch_size=64, 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=5)

    ''' Training procedure '''
    training(train_loader, val_loader, X1.shape[1], X2.shape[1], conf.shape[1])

    ''' Testing procedure '''
    testing(X1, X2, Y, conf)



