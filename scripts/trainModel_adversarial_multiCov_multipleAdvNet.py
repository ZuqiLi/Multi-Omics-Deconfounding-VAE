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
from models.adversarial_XVAE_multiCov_multipleAdvNet import advNet, XVAE_preTrg, XVAE_w_advNet_pingpong
from data.preprocess import ConcatDataset

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
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNAs_processed.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_miRNAs_processed.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
X2 = torch.from_numpy(X2).to(torch.float32)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
# Get traits
Y = traits[:, -1]

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
conf = traits[:, :-1] # stage, age, race, gender

# continous covariates
conf[:,1] = ((conf[:,1] - np.min(conf[:,1])) / (np.max(conf[:,1]) - np.min(conf[:,1]))) 
# Discrete covariates
ohe = OneHotEncoder(sparse_output=False, drop="if_binary")
conf_cat = []
for col in [3,0,2]: # 3,0,2]:
    conf_cat.append(ohe.fit_transform(conf[:,[col]]))
conf_cat_conc = np.concatenate(conf_cat, axis=1)
# Concentenate
conf =  np.concatenate((conf[:,[1]], conf_cat_conc), axis=1)
print("\n",{conf.shape})

''' Dirty dic '''
dic_conf = {"age_CONT":[0], 
            "sex_OHE":[1],
            "stage_OHE":[2,3,4,5],
            "race_OHE":[6,7,8]}

''' Split into training and validation sets '''
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:2100], indices[2100:2700], indices[2700:]
X1_train, X1_val, X1_test = X1[train_idx,:], X1[val_idx,:], X1[test_idx,:]
X2_train, X2_val, X2_test = X2[train_idx,:], X2[val_idx,:], X2[test_idx,:] 
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
outname = "advTraining_multipleAdvNet/advTraining_multiCov"

## Set number of latent features
ls = 50

## pretrainig epochs
epochs_preTrg_ae = 1        #10
epochs_preTrg_advNet = 1   #10

## adversarial training epochs
epochs_ae_w_advNet = [1, 100] #, 100] 

'''
Step 1: pre-train XVAE 
'''
model_pre_XVAE = XVAE_preTrg(X1.shape[1], 
                             X2.shape[1], 
                             ls=ls, 
                             distance='mmd', 
                             beta=1)
ModelSummary(model_pre_XVAE, max_depth=1)

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
    model_xvae_adv = XVAE_w_advNet_pingpong(PATH_xvae_ckpt=ckpt_xvae_file,
                                            PATH_advNet_ckpt=ckpt_advNet_file,
                                            dic_conf = dic_conf,
                                            lamdba_deconf = 1)

    logger_xvae_adv_pingpong = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{outname}/XVAE_adv_pingpong")
    trainer_xvae_adv_pingpong  = L.Trainer(default_root_dir=os.getcwd(), 
                        accelerator="auto", 
                        devices=1, 
                        log_every_n_steps=10, 
                        logger=logger_xvae_adv_pingpong, 
                        max_epochs=epochs,
                        fast_dev_run=False) #
    trainer_xvae_adv_pingpong.fit(model_xvae_adv, train_loader, val_loader)
    trainer_xvae_adv_pingpong.test(dataloaders=test_loader, ckpt_path='best')
    os.rename(f"lightning_logs/{outname}/XVAE_adv_pingpong/version_0", f"lightning_logs/{outname}/XVAE_adv_pingpong/epoch{epochs}")
