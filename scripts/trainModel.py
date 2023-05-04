import os
import numpy as np
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
sys.path.append("./")
from models.autoencoder import XVAE, VAE, AE, init_weights
from data.preprocess import ConcatDataset


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
print(X1.shape, X2.shape)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
Y = traits[:, -1]


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
Y_test = Y[test_idx]

print(X1_test.shape, X2_test.shape, X1[test_idx,:].shape)
print(len(train_idx),len(val_idx),len(test_idx))
print("\n\n")

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
test_loader = data.DataLoader(
                    ConcatDataset(X1_test, X2_test, Y_test), 
                    batch_size=64, 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=5)


model = XVAE(X1.shape[1], X2.shape[1], ls=64, distance='mmd', beta=1)
#print(model)
ModelSummary(model, max_depth=1)

# Initialize Trainer and setting parameters
logger = TensorBoardLogger(save_dir=os.getcwd())
trainer = L.Trainer(default_root_dir=os.getcwd(), accelerator="auto", devices=1, max_epochs=25, log_every_n_steps=10, logger=logger) ##fast_dev_run=True)#
# Use trainer to fit vae model to dataset
trainer.fit(model, train_loader, val_loader)
# automatically auto-loads the best weights from the previous run
trainer.test(dataloaders=test_loader)




