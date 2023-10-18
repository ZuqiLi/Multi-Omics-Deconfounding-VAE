import os 
import sys
import numpy as np
import pytorch_lightning as L
import torch

def set_seed(seed=1234):
    ''' Set seeds for replicability  -Ensure that all operations are deterministic on GPU (if used) for reproducibility '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def load_originalData():

    set_seed()

    ''' Set PATHs '''
    PATH_data = "Data"

    ''' Load data '''
    X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_processed.csv'), delimiter=",")
    X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_processed.csv'), delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
    Y = traits[:, -1]

    return X1, X2, Y


def load_linearConfounder():

    set_seed()

    ''' Set PATHs '''
    PATH_data = "Data"

    ''' Load data '''
    X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded_linear.csv'), delimiter=",")
    X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded_linear.csv'), delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
    Y = traits[:, -1]

    # load artificial confounder
    conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder_linear.csv'))[:,None]
    conf_labels = conf.copy()
    conf = torch.from_numpy(conf).to(torch.float32)
    print('Shape of confounders:', conf.shape)

    return X1, X2, Y, conf, conf_labels



def load_squaredConfounder():

    set_seed()

    ''' Set PATHs '''
    PATH_data = "Data"

    ''' Load data '''
    X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded.csv'), delimiter=",")
    X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded.csv'), delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
    Y = traits[:, -1]

    # load artificial confounder
    conf_type = 'continuous'
    conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder.csv'))[:,None]
    conf_labels = conf.copy()
    conf = torch.from_numpy(conf).to(torch.float32)
    if conf_type == 'categ':
        conf = torch.nn.functional.one_hot(conf[:,0].to(torch.int64))
    print('Shape of confounders:', conf.shape)

    return X1, X2, Y, conf, conf_labels


def load_categoricalConfounder():

    set_seed()

    ''' Set PATHs '''
    PATH_data = "Data"

    ''' Load data '''
    X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded_categ.csv'), delimiter=",")
    X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded_categ.csv'), delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
    Y = traits[:, -1]

    # load artificial confounder
    conf_type = 'categ'
    conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder_categ.csv'))[:,None]
    conf_labels = conf.copy()
    conf = torch.from_numpy(conf).to(torch.float32)
    if conf_type == 'categ':
        conf = torch.nn.functional.one_hot(conf[:,0].to(torch.int64))
    print('Shape of confounders:', conf.shape)

    return X1, X2, Y, conf, conf_labels



def load_categoricalConfounder2():

    set_seed()

    ''' Set PATHs '''
    PATH_data = "Data"

    ''' Load data '''
    X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNA2_confounded_categ2.csv'), delimiter=",")
    X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_DNAm_confounded_categ2.csv'), delimiter=",")
    X1 = torch.from_numpy(X1).to(torch.float32)
    X2 = torch.from_numpy(X2).to(torch.float32)
    traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
    Y = traits[:, -1]

    # load artificial confounder
    conf_type = 'categ'
    conf = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_confounder_categ2.csv'))[:,None]
    conf_labels = conf.copy()
    conf = torch.from_numpy(conf).to(torch.float32)
    if conf_type == 'categ':
        conf = torch.nn.functional.one_hot(conf[:,0].to(torch.int64))
    print('Shape of confounders:', conf.shape)

    return X1, X2, Y, conf, conf_labels