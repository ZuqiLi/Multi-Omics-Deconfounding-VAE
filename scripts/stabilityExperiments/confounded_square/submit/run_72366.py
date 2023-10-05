import os
import numpy as np
import pandas as pd
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
PATH = "/trinity/home/skatz/PROJECTS/Multi-view-Deconfounding-VAE"
sys.path.append(PATH)
from models.clustering import *
from Data.preprocess import *
from models.func import reconAcc_relativeError
### Specify here which model you want to use: XVAE_adversarial_multiclass, XVAE_scGAN_multiclass, XVAE_adversarial_1batch_multiclass
from models.adversarial_XVAE_multiclass import XVAE, advNet, XVAE_scGAN_multiclass 



''' Set seeds for replicability  -Ensure that all operations are deterministic on GPU (if used) for reproducibility '''
np.random.seed(72366)
torch.manual_seed(72366)
L.seed_everything(72366, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' Set PATHs '''
#PATH_data = "Data"
### For EMC cluster
PATH_data = "/data/scratch/skatz/PROJECTS/multiview_VAE/data"

''' Load data '''
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded.csv'), delimiter=",")
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
conf_type = 'continuous'
conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder.csv'))[:,None]
conf = torch.from_numpy(conf).to(torch.float32)             ### Watch out: continous variables should go first
if conf_type == 'categ':
    conf = torch.nn.functional.one_hot(conf[:,0].to(torch.int64))
print('Shape of confounders:', conf.shape)

# specify confounders for advNet training
num_conf_regr = conf.shape[1]         ### None  
num_conf_clf = None            ###conf.shape[1]      
labels_onehot = [f"artificial{i}" for i in range(conf.shape[1])] 
print('\n\n Shape of confounders:', conf.shape[1], "\n\n")


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


#################################################
##             Training procedure              ##
#################################################
modelname = "confounded_square/stability/XVAE_scGAN_multiclass/run_72366" #"confounded_categ2/XVAE_advTraining/XVAE_advTraining_multiclass"
ls = 50

## pretrainig epochs
epochs_preTrg_ae = 5        #10
epochs_preTrg_advNet = 5    #10

## adversarial training epochs
epochs_ae_w_advNet = 150


'''
Step 1: pre-train XVAE 
'''
model_pre_XVAE = XVAE(input_size = [X1.shape[1], X2.shape[1]],
                    # first hidden layer: individual encoding of X1 and X2; [layersizeX1, layersizeX2]; length: number of input modalities
                    hidden_ind_size =[200, 200],
                    # next hidden layer(s): densely connected layers of fused X1 & X2; [layer1, layer2, ...]; length: number of hidden layers
                    hidden_fused_size = [200],                                   
                    ls=ls,                                     
                    distance='mmd',
                    lossReduction='sum', 
                    klAnnealing=False,
                    beta=1,
                    dropout=0.2,
                    init_weights_func="rai")
print(model_pre_XVAE)

# Initialize Trainer
logger_xvae = TensorBoardLogger(save_dir=PATH, name=f"lightning_logs/{modelname}/pre_XVAE")
trainer_xvae = L.Trainer(default_root_dir=PATH, 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger_xvae, 
                    max_epochs=epochs_preTrg_ae,
                    fast_dev_run=False,
                    deterministic=True)
trainer_xvae.fit(model_pre_XVAE, train_loader, val_loader)
os.rename(f"{PATH}/lightning_logs/{modelname}/pre_XVAE/version_0", f"{PATH}/lightning_logs/{modelname}/pre_XVAE/epoch{epochs_preTrg_ae}")


''' 
Step 2: pre-train adv net 
'''
ckpt_xvae_path = f"{PATH}/lightning_logs/{modelname}/pre_XVAE/epoch{epochs_preTrg_ae}/checkpoints"
ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"
model_pre_advNet = advNet(PATH_xvae_ckpt=ckpt_xvae_file,
                          ls=ls, 
                          num_cov_regr=num_conf_regr,
                          num_cov_clf=num_conf_clf)
print("\n\n", model_pre_advNet)

# Initialize Trainer
logger_advNet = TensorBoardLogger(save_dir=PATH, name=f"lightning_logs/{modelname}/pre_advNet")
trainer_advNet = L.Trainer(default_root_dir=PATH, 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger_advNet, 
                    max_epochs=epochs_preTrg_advNet,
                    fast_dev_run=False,
                    deterministic=True)
trainer_advNet.fit(model_pre_advNet, train_loader, val_loader)
os.rename(f"{PATH}/lightning_logs/{modelname}/pre_advNet/version_0", f"{PATH}/lightning_logs/{modelname}/pre_advNet/epoch{epochs_preTrg_advNet}")


''' 
Step 3: train XVAE with adversarial loss in ping pong fashion
'''
ckpt_xvae_path = f"{PATH}/lightning_logs/{modelname}/pre_XVAE/epoch{epochs_preTrg_ae}/checkpoints"
ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"
ckpt_advNet_path = f"{PATH}/lightning_logs/{modelname}/pre_advNet/epoch{epochs_preTrg_advNet}/checkpoints"
ckpt_advNet_file = f"{ckpt_advNet_path}/{os.listdir(ckpt_advNet_path)[0]}"


for epoch in [1, epochs_ae_w_advNet]:

    model_xvae_adv = XVAE_scGAN_multiclass(PATH_xvae_ckpt=ckpt_xvae_file,
                                                PATH_advNet_ckpt=ckpt_advNet_file,
                                                lamdba_deconf = 1,
                                                labels_onehot = labels_onehot)
    print("\n\n", model_xvae_adv)

    logger_xvae_adv = TensorBoardLogger(save_dir=PATH, name=f"lightning_logs/{modelname}/XVAE_adversarialTrg")
    trainer_xvae_adv  = L.Trainer(default_root_dir=PATH, 
                        accelerator="auto", 
                        devices=1, 
                        log_every_n_steps=10, 
                        logger=logger_xvae_adv, 
                        max_epochs=epoch,
                        fast_dev_run=False,
                        deterministic=True) #
    trainer_xvae_adv.fit(model_xvae_adv, train_loader, val_loader)
    os.rename(f"{PATH}/lightning_logs/{modelname}/XVAE_adversarialTrg/version_0", f"{PATH}/lightning_logs/{modelname}/XVAE_adversarialTrg/epoch{epoch}")



###############################################
##         Test on the whole dataset         ##
###############################################
X1_test = scale(X1)
X2_test = scale(X2)
conf_test = conf
conf = conf.detach().numpy()
labels = ['Confounder']

RE_X1s, RE_X2s, RE_X1X2s = [], [], []
clusts = []
SSs, DBs = [], []
n_clust = len(np.unique(Y))
corr_diff = []

## advNet performance for predicting the confounder
from sklearn.metrics import roc_auc_score
scores = np.zeros((50, conf_test.shape[1]))
for i in range(50):         
    ckpt_path = f"{PATH}/lightning_logs/{modelname}/XVAE_adversarialTrg/epoch{epochs_ae_w_advNet}/checkpoints"
    ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"
    ckpt_xvae_path = f"{PATH}/lightning_logs/{modelname}/pre_XVAE/epoch{epochs_preTrg_ae}/checkpoints"
    ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"
    ckpt_advNet_path = f"{PATH}/lightning_logs/{modelname}/pre_advNet/epoch{epochs_preTrg_advNet}/checkpoints"
    ckpt_advNet_file = f"{ckpt_advNet_path}/{os.listdir(ckpt_advNet_path)[0]}"

    model = XVAE_scGAN_multiclass.load_from_checkpoint(ckpt_file,
            PATH_xvae_ckpt=ckpt_xvae_file, PATH_advNet_ckpt=ckpt_advNet_file, map_location=torch.device("cpu"))
    
    # Loop over dataset and test on batches
    indices = np.array_split(np.arange(X1_test.shape[0]), 20)
    y_pred_all, y_true_all = [], []
    for idx in indices:
        _, y_pred = model.advNet_pre.forward(X1_test[idx], X2_test[idx])
        y_pred = [y_pred[j].detach().numpy() for j in range(y_pred.shape[0])]
        y_pred_all.append(y_pred)

    y_pred_all = np.concatenate(y_pred_all)

    for j in range(conf_test.shape[1]):
        scores[i,j] = roc_auc_score(conf_test[:,j], y_pred_all[:,j])

scores = np.mean(scores, 0)
for i in range(conf_test.shape[1]):
    print(f"Conf{i+1}", "\t", round(scores[i], 2))

## Compute reconstruction and clustering metrics
# Sample multiple times from the latent distribution for stability
for i in range(50):         
    corr_res = []
    for epoch in [1, epochs_ae_w_advNet]:
        ckpt_path = f"{PATH}/lightning_logs/{modelname}/XVAE_adversarialTrg/epoch{epoch}/checkpoints"
        ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"
        ckpt_xvae_path = f"{PATH}/lightning_logs/{modelname}/pre_XVAE/epoch{epochs_preTrg_ae}/checkpoints"
        ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"
        ckpt_advNet_path = f"{PATH}/lightning_logs/{modelname}/pre_advNet/epoch{epochs_preTrg_advNet}/checkpoints"
        ckpt_advNet_file = f"{ckpt_advNet_path}/{os.listdir(ckpt_advNet_path)[0]}"

        model = XVAE_scGAN_multiclass.load_from_checkpoint(ckpt_file,
                PATH_xvae_ckpt=ckpt_xvae_file, PATH_advNet_ckpt=ckpt_advNet_file, map_location=torch.device("cpu"))   #### addition so it works with CUDS as well
        
        # Loop over dataset and test on batches
        indices = np.array_split(np.arange(X1_test.shape[0]), 20)
        z = []
        X1_hat, X2_hat = [], []
        for idx in indices:
            z_batch = model.xvae.generate_embedding(X1_test[idx], X2_test[idx])
            z.append(z_batch.detach().numpy())
            X1_hat_batch, X2_hat_batch = model.xvae.decode(z_batch)
            X1_hat.append(X1_hat_batch.detach().numpy())
            X2_hat.append(X2_hat_batch.detach().numpy())

        z = np.concatenate(z)
        X1_hat = np.concatenate(X1_hat)
        X2_hat = np.concatenate(X2_hat)

        if epoch == epochs_ae_w_advNet:
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
        
        # Correlation between latent vectors and the confounder
        corr_conf = [np.abs(np.corrcoef(z.T, conf[:,i])[:-1,-1]) for i in range(conf.shape[1])]
        if conf_type == 'categ':
            corr_conf = [np.mean(corr_conf)]
        corr_res.append(pd.DataFrame(corr_conf, index=labels))
    # Calculate correlation difference
    # (corr_first_epoch - corr_last_epoch) / corr_first_epoch
    corr_diff.append(list(((corr_res[0].T - corr_res[1].T).mean() / corr_res[0].T.mean())*100))

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
ARI, NMI = external_metrics(con_clust, Y)
print("ARI for cancer types:", ARI)
print("NMI for cancer types:", NMI)
if conf_type == 'categ':
    conf = np.argmax(conf, 1)
else:
    conf = conf[:,0]
ARI_conf, NMI_conf = external_metrics(con_clust, conf)
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

pd.DataFrame(res).to_csv(f"{PATH}/lightning_logs/{modelname}/XVAE_adversarialTrg/epoch{epochs_ae_w_advNet}/results_performance.csv", index=False)

