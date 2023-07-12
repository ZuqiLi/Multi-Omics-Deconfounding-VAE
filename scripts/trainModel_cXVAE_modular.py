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
from models.cXVAE import cXVAE_fusedEmbed
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


''' Split into training and validation sets '''
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:2100], indices[2100:2700], indices[2700:]

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

print([X1.shape[1], X2.shape[1]])

modelname = 'cVAE_modular/cXVAE_mod'
maxEpoch = 2

for epoch in [1, maxEpoch]:
    model = cXVAE_fusedEmbed(input_size=[X1.shape[1], X2.shape[1]],
                    hidden_ind_size =[200, 200],                ### first hidden layer: individual encoding of X1 and X2; [layersizeX1, layersizeX2]; length: number of input modalities
                    hidden_fused_size = [200],                  ### next hidden layer(s): densely connected layers of fused X1 & X2; [layer1, layer2, ...]; length: number of hidden layers
                    ls=50,                                      ### latent size
                    cov_size=conf.shape[1],                                   
                    distance='mmd',
                    lossReduction='sum', 
                    klAnnealing=False,
                    beta=1,
                    dropout=0.2,
                    init_weights_func="rai")
    
    print(model)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"lightning_logs/{modelname}/")
    trainer = L.Trainer(default_root_dir=os.getcwd(), 
                        accelerator="auto", 
                        devices=1, 
                        log_every_n_steps=10, 
                        logger=logger, 
                        max_epochs=epoch,
                        fast_dev_run=False)
    #trainer.fit(model, train_loader, val_loader)
    #trainer.test(dataloaders=test_loader, ckpt_path='best')
    #os.rename(f"lightning_logs/{modelname}/version_0", f"lightning_logs/{modelname}/epoch{epoch}")



#############################################################
##        calculate corr coefficient difference            ##
#############################################################

labels_onehot = ['Confounder']
#labels_onehot = ['Gender', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Age', 'Race1', 'Race2', 'Race3']
# Because of the variational part the latent space is always a bit different and these values change
all_corr = []
for i in range(1):  ## 50
    res = []
    for epoch in [1, 2]:       ## 100
        ckpt_path = f"{os.getcwd()}/lightning_logs/{modelname}/epoch{maxEpoch}/checkpoints"
        ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"

        model = cXVAE_fusedEmbed.load_from_checkpoint(ckpt_file)
        z = model.generate_embedding(X1_test, X2_test, conf_test).detach().numpy()

        conf_test = conf_test.detach().clone()
        #bins = conf[:, 5:15]
        #digits = np.argmax(bins, axis=1)
        #conf = np.concatenate((conf[:,:5], digits[:,None], conf[:,15:]), axis=1)
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
print(corr_dict)


#############################################################
##        Compute consensus clsutering and metrics         ##
#############################################################

labels = []
SSs, DBs = [], []
n_clust = len(np.unique(Y))
for i in range(2):
    ckpt_path = f"{os.getcwd()}/lightning_logs/{modelname}/epoch{maxEpoch}/checkpoints"
    ckpt_file = f"{ckpt_path}/{os.listdir(ckpt_path)[0]}"

    model = cXVAE_fusedEmbed.load_from_checkpoint(ckpt_file)
    z = model.generate_embedding(scale(X1), scale(X2), conf).detach().numpy()

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
ARI_conf, NMI_conf = external_metrics(con_clust, conf[:,0])
print("ARI for confounder:", ARI_conf)
print("NMI for confounder:", NMI_conf)



