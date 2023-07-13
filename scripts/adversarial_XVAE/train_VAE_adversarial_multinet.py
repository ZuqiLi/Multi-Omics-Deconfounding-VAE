import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
sys.path.append("./")
from models.adversarial_XVAE_multipleAdvNet_modular import advNet, XVAE, XVAE_adversarial_multinet
from Data.preprocess import ConcatDataset, scale
from models.clustering import *
from models.func import reconAcc_pearsonCorr, reconAcc_relativeError


''' 
XVAE with adversarial training as deconfounding strategy

[*] Step 1: pre-train XVAE 
[*] Step 2: pre-train adversarial net from latent embedding from Step 1
[*] Step 3: fix advNet and train XVAE with combined loss
'''

''' Set seeds for replicability 
Ensure that all operations are deterministic on GPU (if used) for reproducibility 
'''
np.random.seed(1234)
torch.manual_seed(1234)
L.seed_everything(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' Set PATHs '''
PATH_data = "Data"

''' Load data '''
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
X2 = torch.from_numpy(X2).to(torch.float32)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
# Get traits
Y = traits[:, -1]
artificialConf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder.csv'), delimiter=",")[:,None]

''' 
Specify confounders 

1. specify which columns
2. scale continuous variables
    - if only 1 continuous variable: transform to matrix with 1 column 
3. OHE discrete variables
4. Set not used variables types to `None`:
    - if no continous confounders: `num_conf_regr` = None 
    - if no discrete confounders: `num_conf_discrete` = None 
'''
conf = OneHotEncoder(sparse_output=False, drop="if_binary").fit_transform(artificialConf)
conf = torch.from_numpy(conf).to(torch.float32)
print("\n",{conf.shape})

''' Dirty dic '''
dic_conf = {"artificialConf_OHE":[0,1,2,3,4,5]}

''' Split into training and validation sets '''
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:1600], indices[1600:2100], indices[2100:]

X1_train, X1_val, X1_test = scale(X1[train_idx,:]), scale(X1[val_idx,:]), scale(X1[test_idx,:])
X2_train, X2_val, X2_test = scale(X2[train_idx,:]), scale(X2[val_idx,:]), scale(X2[test_idx,:])
conf_train, conf_val, conf_test = conf[train_idx,:], conf[val_idx,:], conf[test_idx,:]
Y_test = Y[test_idx]

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
test_loader = data.DataLoader(
                    ConcatDataset(X1_test, X2_test, conf_test, Y_test), 
                    batch_size=64, 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=5)

'''
#################################################
##             Training procedure              ##
#################################################
''' 

''' 
Step 0: settings
'''

## Name of the folder
outname = "adversarialTrg/modularTest/multinet"

## Set number of latent features
ls = 50

## pretrainig epochs
epochs_preTrg_ae = 1        #10
epochs_preTrg_advNet = 1   #10

## adversarial training epochs
epochs_ae_w_advNet = [1, 2] #, 100] 

'''
Step 1: pre-train XVAE 
'''
model_pre_XVAE = XVAE(input_size = [X1.shape[1], X2.shape[1]],
                            hidden_ind_size =[200, 200],                ### first hidden layer: individual encoding of X1 and X2; [layersizeX1, layersizeX2]; length: number of input modalities
                            hidden_fused_size = [200],                  ### next hidden layer(s): densely connected layers of fused X1 & X2; [layer1, layer2, ...]; length: number of hidden layers
                            ls=ls,                                      ### latent size
                            distance='mmd',
                            lossReduction='sum', 
                            klAnnealing=False,
                            beta=1,
                            dropout=0.2,
                            init_weights_func="rai")
print(model_pre_XVAE)

# Initialize Trainer and setting parameters
logger_xvae = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{outname}/pre_XVAE")
trainer_xvae  = L.Trainer(default_root_dir=os.getcwd(), 
                          accelerator="auto", 
                          devices=1, 
                          log_every_n_steps=10, 
                          logger=logger_xvae, 
                          max_epochs=epochs_preTrg_ae, 
                          fast_dev_run=False) #
trainer_xvae.fit(model_pre_XVAE, train_loader, val_loader)
trainer_xvae.test(dataloaders=test_loader, ckpt_path='best')
os.rename(f"lightning_logs/{outname}/pre_XVAE/version_0", f"lightning_logs/{outname}/pre_XVAE/epoch{epochs_preTrg_ae}")


''' 
Step 2: pre-train adv net 
'''
ckpt_xvae_path = f"{os.getcwd()}/lightning_logs/{outname}/pre_XVAE/epoch{epochs_preTrg_ae}/checkpoints"
ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"

model_pre_advNet = advNet(PATH_xvae_ckpt=ckpt_xvae_file,
                          ls=ls, 
                          dic_conf = dic_conf)

logger_advNet = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{outname}/pre_advNet")
trainer_advNet  = L.Trainer(default_root_dir=os.getcwd(), 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger_advNet, 
                    max_epochs=epochs_preTrg_advNet, 
                    fast_dev_run=False) #
trainer_advNet.fit(model_pre_advNet, train_loader, val_loader)
trainer_advNet.test(dataloaders=test_loader, ckpt_path='best')
os.rename(f"lightning_logs/{outname}/pre_advNet/version_0", f"lightning_logs/{outname}/pre_advNet/epoch{epochs_preTrg_advNet}")


''' 
Step 3: train XVAE with adversarial loss in ping pong fashion
'''
ckpt_xvae_path = f"{os.getcwd()}/lightning_logs/{outname}/pre_XVAE/epoch{epochs_preTrg_ae}/checkpoints"
ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"
ckpt_advNet_path = f"{os.getcwd()}/lightning_logs/{outname}/pre_advNet/epoch{epochs_preTrg_advNet}/checkpoints"
ckpt_advNet_file = f"{ckpt_advNet_path}/{os.listdir(ckpt_advNet_path)[0]}"

for epochs in epochs_ae_w_advNet:
    model_xvae_adv = XVAE_adversarial_multinet(PATH_xvae_ckpt=ckpt_xvae_file,
                                            PATH_advNet_ckpt=ckpt_advNet_file,
                                            dic_conf = dic_conf,
                                            lamdba_deconf = 1)

    logger_xvae_adv_pingpong = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{outname}/XVAE_adversarialTrg")
    trainer_xvae_adv_pingpong  = L.Trainer(default_root_dir=os.getcwd(), 
                        accelerator="auto", 
                        devices=1, 
                        log_every_n_steps=10, 
                        logger=logger_xvae_adv_pingpong, 
                        max_epochs=epochs,
                        fast_dev_run=False) #
    trainer_xvae_adv_pingpong.fit(model_xvae_adv, train_loader, val_loader)
    trainer_xvae_adv_pingpong.test(dataloaders=test_loader, ckpt_path='best')
    os.rename(f"lightning_logs/{outname}/XVAE_adversarialTrg/version_0", f"lightning_logs/{outname}/XVAE_adversarialTrg/epoch{epochs}")




##################################################
##                Reconstruction               ##
#################################################
print("\n\nCompute reconstruction accuracy...\n\n ")

ckpt_path = f"{os.getcwd()}/lightning_logs/{outname}/XVAE_adversarialTrg/epoch{epochs_ae_w_advNet[-1]}/checkpoints"
ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"
model = XVAE_adversarial_multinet.load_from_checkpoint(ckpt_file)

x1_hat, x2_hat = model.xvae.forward(X1_test, X2_test)
x1_hat = x1_hat.detach().numpy()
x2_hat = x2_hat.detach().numpy()

''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
reconAcc_x1 = reconAcc_pearsonCorr(X1_test, x1_hat)
reconAcc_x2 = reconAcc_pearsonCorr(X2_test, x2_hat)
                                   
''' Relative Error using L2 norm '''
relativeError = reconAcc_relativeError(X1_test, x1_hat,  X2_test, x2_hat)

print(f'Reconstruction accuracy X1 - Pearson correlation (mean+-std)   \t {np.mean(reconAcc_x1):.3f}+-{np.std(reconAcc_x1):.3f}')
print(f'Reconstruction accuracy X2 - Pearson correlation (mean+-std)   \t {np.mean(reconAcc_x2):.3f}+-{np.std(reconAcc_x2):.3f}')
print(f'Reconstruction accuracy - Relative error (L2 norm)   \t {relativeError:.3f} \n\n')


#############################################################
##        calculate corr coefficient difference            ##
#############################################################
# Because of the variational part the latent space is always a bit different and these values change


labels_onehot = [f'Confounder{i}' for i in range(1,conf.shape[1]+1)]

all_corr = []
for i in range(1):  ## 50
    res = []
    for epoch in epochs_ae_w_advNet:       ## 100
        ckpt_path = f"{os.getcwd()}/lightning_logs/{outname}/XVAE_adversarialTrg/epoch{epochs_ae_w_advNet[-1]}/checkpoints"
        ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"

        model = XVAE_adversarial_multinet.load_from_checkpoint(ckpt_file)
        z = model.xvae.generate_embedding(X1_test, X2_test).detach().numpy()

        conf_test = conf_test.detach().clone()
        corr_conf = [np.abs(np.corrcoef(z.T, conf_test[:,i].T)[:-1,-1]) for i in range(conf_test.shape[1])]
        res.append(pd.DataFrame(corr_conf, index=labels_onehot))
    ''' 
    Calculate [%] differences of correlation of confounders to each latent feature before (epoch1) and after training (epoch100)

    Formula:
    ((first epoch) - (last epoch)).mean() / (first epoch).mean()

    '''
    all_corr.append(list(((res[0].T - res[1].T).mean() / res[0].T.mean())*100))

# Average over all samplings
all_corr_unpacked = list(zip(*all_corr))
corr_dict = dict()
for i, label in enumerate(labels_onehot):
    corr_dict[label] = np.array(all_corr_unpacked[i]).mean()
print("\n\n", corr_dict, "\n\n")



##################################################
#   Compute consensus clustering and metrics     #
##################################################
print("\n\nCompute clustering...\n\n ")

labels = []
SSs, DBs = [], []
n_clust = len(np.unique(Y))
for i in range(1):      ### 20
    ckpt_path = f"{os.getcwd()}/lightning_logs/{outname}/XVAE_adversarialTrg/epoch{epochs_ae_w_advNet[-1]}/checkpoints"
    ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"
    model = XVAE_adversarial_multinet.load_from_checkpoint(ckpt_file)

    z = model.xvae.generate_embedding(scale(X1), scale(X2)).detach().numpy()

    label = kmeans(z, n_clust)
    labels.append(label)
    
    SS, DB = internal_metrics(z, label)
    SSs.append(SS)
    DBs.append(DB)

con_clust, _, disp = consensus_clustering(labels, n_clust)
print("Dispersion for co-occurrence matrix:", disp)

print("Silhouette score:", np.mean(SSs))
print("DB index:", np.mean(DBs))
ARI, NMI = external_metrics(con_clust, Y)
print("ARI for cancer types:", round(ARI, 3))
print("NMI for cancer types:", round(NMI, 3))
print('\n\n')
ARI_conf, NMI_conf = external_metrics(con_clust, conf[:,0])
print("ARI for confounder:", round(ARI_conf, 3))
print("NMI for confounder:", round(NMI_conf, 3))
print('\n\n')

##################################################
#                  Save to file                  #
##################################################

### Save to file
res = {'recon_x1':[np.mean(reconAcc_x1)],
    'recon_x2':[np.mean(reconAcc_x2)],
    'l2_error':[relativeError],
    'deconfounding_correlationCoef':[list(corr_dict.values())],
    'CC_dispersion':[disp],
    'ss':[np.mean(SSs)],
    'db':[np.mean(DBs)],
    'ari_trueCluster':[ARI],
    'nmi_trueCluster':[NMI],
    'ari_confoundedCluster':[ARI],
    'nmi_confoundedCluster':[NMI]
    }

pd.DataFrame(res).to_csv(f"lightning_logs/{outname}/XVAE_adversarialTrg/epoch{epochs_ae_w_advNet[-1]}/results_performance.csv")