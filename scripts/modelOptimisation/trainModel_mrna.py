import os
import numpy as np
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
sys.path.append("./")
from models.vanilla_autoencoder import VAE
from Data.preprocess import ConcatDataset, scale
from models.clustering import *
from models.func import reconAcc_pearsonCorr, reconAcc_relativeError


''' Set seeds for replicability  -Ensure that all operations are deterministic on GPU (if used) for reproducibility '''
np.random.seed(1234)
torch.manual_seed(1234)
L.seed_everything(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' Set PATHs '''
PATH_data = "Data"


''' Load data '''
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_processed.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
Y = traits[:, -1]


''' Split into training and validation sets '''
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:1875], indices[1875:2100], indices[2100:]

X1_train, X1_val, X1_test = scale(X1[train_idx,:]), scale(X1[val_idx,:]), scale(X1[test_idx,:])
Y_test = Y[test_idx]

''' Initialize Dataloader '''
train_loader = data.DataLoader(
                    X1_train, 
                    batch_size=64, 
                    shuffle=True, 
                    drop_last=False, 
                    num_workers=5)
val_loader = data.DataLoader(
                    X1_val, 
                    batch_size=64, 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=5)
test_loader = data.DataLoader(
                    ConcatDataset(X1_test, Y_test), 
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
outname = "singleOmic/mRNA"
maxEpochs = 51

model = VAE(x_size=X1.shape[1],
            hidden_layer_encoder_topology=[500, 200],
            ls=50, 
            distance='mmd', 
            beta=1)
ModelSummary(model, max_depth=1)
print(model)

# Initialize Trainer and setting parameters
logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{outname}")
trainer = L.Trainer(default_root_dir=os.getcwd(), 
                    accelerator="auto", 
                    devices=1, 
                    max_epochs=maxEpochs, 
                    log_every_n_steps=10, 
                    logger=logger,
                    fast_dev_run=False)
# Use trainer to fit vae model to dataset
trainer.fit(model, train_loader, val_loader)
# automatically auto-loads the best weights from the previous run
trainer.test(dataloaders=test_loader)
os.rename(f"lightning_logs/{outname}/version_0", f"lightning_logs/{outname}/epoch{maxEpochs}")



##################################################
##                Reconstruction               ##
##################################################
print("\n\nCompute reconstruction accuracy...\n\n ")

ckpt_path = f"{os.getcwd()}/lightning_logs/{outname}/epoch{maxEpochs}/checkpoints"
ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"
model = VAE.load_from_checkpoint(ckpt_file)

x1_hat= model.forward(X1_test)
x1_hat = x1_hat.detach().numpy()

''' Reconstruction accuracy (Pearson correlation between reconstruction and input) '''
reconAcc_x1 = reconAcc_pearsonCorr(X1_test, x1_hat)
                                   

print(f'Reconstruction accuracy X1 - Pearson correlation (mean+-std)   \t {np.mean(reconAcc_x1):.3f}+-{np.std(reconAcc_x1):.3f}')

##################################################
#   Compute consensus clustering and metrics
##################################################
print("\n\nCompute clustering...\n\n ")

labels = []
SSs, DBs = [], []
n_clust = len(np.unique(Y))
for i in range(20):
    ckpt_path = f"{os.getcwd()}/lightning_logs/{outname}/epoch{maxEpochs}/checkpoints"
    ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"
    model = VAE.load_from_checkpoint(ckpt_file)

    z = model.generate_embedding(scale(X1)).detach().numpy()

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
print("ARI for cancer types:", ARI)
print("NMI for cancer types:", NMI)
print('\n\n')
# ARI_conf, NMI_conf = external_metrics(con_clust, conf[:,0])
# print("ARI for confounder:", ARI_conf)
# print("NMI for confounder:", NMI_conf)

