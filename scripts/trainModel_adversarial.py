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
from models.adversarial_XVAE import advNet, XVAE_preTrg, XVAE_w_advNet
from data.preprocess import ConcatDataset

''' 
XVAE with adversarial training as deconfounding strategy

[*] Step 1: pre-train XVAE 
[*] Step 2: pre-train adversarial net from latent embedding from Step 1
[*] Step 3: fix advNet and train XVAE with combined loss
'''

''' Set seeds for replicability  -Ensure that all operations are deterministic on GPU (if used) for reproducibility '''
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
# Take only age as confounder and scale
conf = traits[:, 1] 
conf = (conf - np.min(conf)) / (np.max(conf) - np.min(conf))
print('Shape of confounders:', conf.shape)

''' Split into training and validation sets '''
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:2100], indices[2100:2700], indices[2700:]

##### I am not a big fan of that as we also want to test other metrices using the test set... let's remove it for now and check how to implement it...
# # we test on the whole dataset for clustering
# train_idx = np.concatenate((train_idx, test_idx))
# test_idx = indices
X1_train, X1_val, X1_test = X1[train_idx,:], X1[val_idx,:], X1[test_idx,:]
X2_train, X2_val, X2_test = X2[train_idx,:], X2[val_idx,:], X2[test_idx,:] 
conf_train, conf_val, conf_test = conf[train_idx,], conf[val_idx,], conf[test_idx,] 
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
Training procedure 
'''

ls = 50

''' Step 1: pre-train XVAE '''
model_pre_XVAE = XVAE_preTrg(X1.shape[1], 
              X2.shape[1], 
              ls=ls, 
              distance='mmd', 
              beta=1)
ModelSummary(model_pre_XVAE, max_depth=1)

# Initialize Trainer and setting parameters
logger_xvae = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs/advTraining/pre_XVAE")
trainer_xvae  = L.Trainer(default_root_dir=os.getcwd(), 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger_xvae, 
                    max_epochs=100,
                    fast_dev_run=True) #
#trainer_xvae.fit(model_pre_XVAE, train_loader, val_loader)
#trainer_xvae.test(dataloaders=test_loader)


''' Step 2: pre-train adv net '''
ckpt_xvae_path = f"{os.getcwd()}/lightning_logs/advTraining/pre_XVAE/version_0/checkpoints"
ckpt_xvae_file = f"{ckpt_xvae_path}/{os.listdir(ckpt_xvae_path)[0]}"

model_pre_advNet = advNet(PATH_xvae_ckpt=ckpt_xvae_file,
                          ls=ls, 
                          cov_size=1)

logger_advNet = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs/advTraining/pre_advNet")
trainer_advNet  = L.Trainer(default_root_dir=os.getcwd(), 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger_advNet, 
                    max_epochs=100,
                    fast_dev_run=True) #
#trainer_advNet.fit(model_pre_advNet, train_loader, val_loader)
#trainer_advNet.test(dataloaders=test_loader)

''' Step 3: train XVAE with adversarial loss '''
ckpt_advNet_path = f"{os.getcwd()}/lightning_logs/advTraining/pre_advNet/version_0/checkpoints"
ckpt_advNet_file = f"{ckpt_advNet_path}/{os.listdir(ckpt_advNet_path)[0]}"

model_xvae_adv = XVAE_w_advNet(PATH_xvae_ckpt=ckpt_xvae_file,
                               PATH_advNet_ckpt=ckpt_advNet_file)

logger_xvae_adv = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs/advTraining/XVAE_adv")
trainer_xvae_adv  = L.Trainer(default_root_dir=os.getcwd(), 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger_xvae_adv, 
                    max_epochs=100,
                    fast_dev_run=False) #
trainer_xvae_adv.fit(model_xvae_adv, train_loader, val_loader)
trainer_xvae_adv.test(dataloaders=test_loader)