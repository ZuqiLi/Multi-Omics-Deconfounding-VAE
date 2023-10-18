import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
PATH = "/trinity/home/skatz/PROJECTS/Multi-view-Deconfounding-VAE"
sys.path.append(PATH)
from models.XVAE import XVAE
from models.clustering import *
from Data.preprocess import *
from models.func import reconAcc_relativeError
from scipy.stats import pearsonr


''' Set seeds for replicability  -Ensure that all operations are deterministic on GPU (if used) for reproducibility '''
np.random.seed(77538)
torch.manual_seed(77538)
L.seed_everything(77538, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' Set PATHs '''
##PATH_data = "Data"
### For EMC cluster
PATH_data = "/data/scratch/skatz/PROJECTS/multiview_VAE/data"


''' Load data '''
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded_multi.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded_multi.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
X2 = torch.from_numpy(X2).to(torch.float32)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
Y = traits[:, -1]
'''
# The rest as confounders
conf = traits[:, :-1] # stage, age, race, gender
age = conf[:,1].copy()
# rescale age to [0,1)
age = (age - np.min(age)) / (np.max(age) - np.min(age) + 1e-8)
# bin age accoring to quantiles
#n_bins = 10
#bins = np.histogram(age, bins=10, range=(age.min(), age.max()+1e-8))[1]
#age = np.digitize(age, bins) # starting from 1
conf[:,1] = age
# onehot encoding
conf_onehot = OneHotEncoder(sparse=False).fit_transform(conf[:,:3])
conf = np.concatenate((conf[:,[3]], conf_onehot), axis=1)
# select only gender
conf = conf[:,[0]]
'''
# load artificial confounder
conf_type = 'multi'
conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder_multi.csv'), delimiter=",")
conf = torch.from_numpy(conf).to(torch.float32)
n_confs = conf.shape[1] if conf.ndim > 1 else 1
if conf_type == 'conti':
    conf = conf[:, None]
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
                    ConcatDataset(X1_train, X2_train), 
                    batch_size=64, 
                    shuffle=True, 
                    drop_last=False, 
                    num_workers=5)
val_loader = data.DataLoader(
                    ConcatDataset(X1_val, X2_val), 
                    batch_size=64, 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=5)


#################################################
##             Training procedure              ##
#################################################
PATH_model =  "/data/scratch/skatz/PROJECTS/multiview_VAE/"
modelname = 'confounded_multi/stability/cutoff_corr_05/run_77538'
maxEpochs = 150

###############################################
##         Test on the whole dataset         ##
###############################################
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
###########################################
for i in range(50): 
    corr_res = []
    for epoch in [1, maxEpochs]:
        ckpt_path = f"{PATH_model}/lightning_logs/{modelname}/epoch{epoch}/checkpoints"
        ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"

        model = XVAE.load_from_checkpoint(ckpt_file, map_location=torch.device('cpu'))
        
        # Loop over dataset and test on batches
        indices = np.array_split(np.arange(X1_test.shape[0]), 20)
        z = []
        X1_hat, X2_hat = [], []
        for idx in indices:
            z_batch = model.generate_embedding(X1_test[idx], X2_test[idx])
            z.append(z_batch.detach().numpy())
            X1_hat_batch, X2_hat_batch = model.decode(z_batch)
            X1_hat.append(X1_hat_batch.detach().numpy())
            X2_hat.append(X2_hat_batch.detach().numpy())

        z = np.concatenate(z)
        X1_hat = np.concatenate(X1_hat)
        X2_hat = np.concatenate(X2_hat)

        if epoch == maxEpochs:
            # Compute relative error from the last epoch
            RE_X1, RE_X2, RE_X1X2 = reconAcc_relativeError(X1_test, X1_hat, X2_test, X2_hat)
            RE_X1s.append(RE_X1)
            RE_X2s.append(RE_X2)
            RE_X1X2s.append(RE_X1X2)

            if z.shape[1] != 0: # not all latent vectors are removed
                # Clustering the latent vectors from the last epoch
                clust = kmeans(z, n_clust)
                clusts.append(clust)
                # Compute clustering metrics
                SS, DB = internal_metrics(z, clust)
                SSs.append(SS)
                DBs.append(DB)
        
        # Correlation between latent vectors and the confounder
        if z.shape[1] != 0: # not all latent vectors are removed
            corr_conf = [np.abs(np.corrcoef(z.T, conf[:,i])[:-1,-1]) for i in range(conf.shape[1])]
        if conf_type == 'categ':
            # take the mean of one-hot variables of the categorical confounder
            corr_conf = [np.mean(corr_conf, axis=0)]
        elif conf_type == 'multi':
            # take the mean of one-hot variables of the categorical confounder
            corr_conf_categ = np.mean(corr_conf[(n_confs-1):], axis=0)
            corr_conf = corr_conf[:(n_confs-1)] + [corr_conf_categ]
        corr_res.append(pd.DataFrame(corr_conf, index=labels))
    # Calculate correlation difference
    # (corr_first_epoch - corr_last_epoch) / corr_first_epoch
    corr_diff.append(list(((corr_res[0].T.mean() - corr_res[1].T.mean()) / corr_res[0].T.mean())*100))

# Average relative errors over all samplings
print("Relative error (X1):", np.mean(RE_X1s))
print("Relative error (X2):", np.mean(RE_X2s))
print("Relative error (X1X2):", np.mean(RE_X1X2s))

# Average correlation differences over all samplings
corr_diff_unpacked = list(zip(*corr_diff))
corr_dict = dict()
for i, label in enumerate(labels):
    corr_dict[label] = np.array(corr_diff_unpacked[i]).mean()
print("Corr diff:", corr_dict)

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
    'deconfounding_corrcoef':[list(corr_dict.values())],
    'CC_dispersion':[disp],
    'ss':[np.mean(SSs)],
    'db':[np.mean(DBs)],
    'ari_trueCluster':[ARI],
    'nmi_trueCluster':[NMI],
    'ari_confoundedCluster':[ARI_conf],
    'nmi_confoundedCluster':[NMI_conf]
    }

pd.DataFrame(res).to_csv(f"{PATH_model}/lightning_logs/{modelname}/epoch{maxEpochs}/results_performance_vanillaXVAE.csv", index=False)

