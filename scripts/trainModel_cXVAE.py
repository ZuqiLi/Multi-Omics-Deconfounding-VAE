import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch 
import pytorch_lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data
import sys
sys.path.append("./")
from models.cXVAE import cXVAE
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
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNAs_confounded.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_miRNAs_confounded.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
X2 = torch.from_numpy(X2).to(torch.float32)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
n_samples = X1.shape[0]
# Get traits
Y = traits[:, -1]
# The rest as confounders  --- TO DO: one hot encode / bin them 
conf = traits[:, :-1] # stage, age, race, gender
age = conf[:,1].copy()
# rescale age to [0,1)
#age = (age - np.min(age)) / (np.max(age) - np.min(age) + 1e-8)
# bin age accoring to quantiles
n_bins = 10
bins = np.histogram(age, bins=10, range=(age.min(), age.max()+1e-8))[1]
age = np.digitize(age, bins) # starting from 1
conf[:,1] = age
# onehot encoding
conf_onehot = OneHotEncoder(sparse=False).fit_transform(conf[:,:3])
conf = np.concatenate((conf[:,[3]], conf_onehot), axis=1)
conf = conf[:,[0]] # select only gender
# load artificial confounder
conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder.csv'))[:,None]
conf = torch.from_numpy(conf).to(torch.float32)
print('Shape of confounders:', conf.shape)


def scale(X):
    '''Min-max normalization to [0,1] along columns'''
    X_min, _ = torch.min(X, dim=0, keepdim=True)
    X_max, _ = torch.max(X, dim=0, keepdim=True)
    X = (X - X_min) / (X_max - X_min)
    return X

''' Split into training and validation sets '''
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:2100], indices[2100:2700], indices[2700:]

##### I am not a big fan of that as we also want to test other metrices using the test set... let's remove it for now and check how to implement it...
# # we test on the whole dataset for clustering
# train_idx = np.concatenate((train_idx, test_idx))
# test_idx = indices
X1_train, X1_val, X1_test = scale(X1[train_idx,:]), scale(X1[val_idx,:]), scale(X1[test_idx,:])
X2_train, X2_val, X2_test = scale(X2[train_idx,:]), scale(X2[val_idx,:]), scale(X2[test_idx,:])
conf_train, conf_val, conf_test = scale(conf[train_idx,:]), scale(conf[val_idx,:]), scale(conf[test_idx,:])
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


modelname = 'cXVAE_conf0'

for max_epochs in [1, 100]:
    model = cXVAE(X1.shape[1], X2.shape[1], ls=64, 
              cov_size=conf.shape[1], distance='mmd', beta=1)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{modelname}/")
    trainer = L.Trainer(default_root_dir=os.getcwd(), 
                        accelerator="auto", 
                        devices=1, 
                        log_every_n_steps=10, 
                        logger=logger, 
                        max_epochs=max_epochs,
                        fast_dev_run=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    os.rename(f"lightning_logs/{modelname}/version_0", f"lightning_logs/{modelname}/epoch{max_epochs}")

##########
### calculate corr coefficient difference
##########
labels_onehot = ['Confounder']
#labels_onehot = ['Gender', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Age', 'Race1', 'Race2', 'Race3']
# Because of the variational part the latent space is always a bit different and these values change
all_corr = []
for i in range(50):
    res = []
    for epoch in [1, 100]:
        ckpt_path = f"{os.getcwd()}/lightning_logs/{modelname}/epoch{epoch}/checkpoints"
        ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"

        model = cXVAE.load_from_checkpoint(ckpt_file)
        z = model.generate_embedding(X1_test, X2_test, conf_test).detach().numpy()

        conf = conf_test.detach().clone()
        #bins = conf[:, 5:15]
        #digits = np.argmax(bins, axis=1)
        #conf = np.concatenate((conf[:,:5], digits[:,None], conf[:,15:]), axis=1)
        corr_conf = [np.abs(np.corrcoef(z.T, conf[:,i].T)[:-1,-1]) for i in range(conf.shape[1])]
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
print(corr_dict)



exit()
model = cXVAE(X1.shape[1], 
              X2.shape[1], 
              ls=64, 
              cov_size=conf.shape[1], 
              distance='mmd', 
              beta=1)
#print(model)
ModelSummary(model, max_depth=1)

# Initialize Trainer and setting parameters
logger = TensorBoardLogger(save_dir=os.getcwd())
trainer = L.Trainer(default_root_dir=os.getcwd(), 
                    accelerator="auto", 
                    devices=1, 
                    log_every_n_steps=10, 
                    logger=logger, 
                    max_epochs=100,
                    fast_dev_run=False) #
# Use trainer to fit vae model to dataset
trainer.fit(model, train_loader, val_loader)
# automatically auto-loads the best weights from the previous run
trainer.test(dataloaders=test_loader)




