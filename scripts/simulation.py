import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1234)

path = '/usr/local/micapollo01/MIC/DATA/STAFF/zli1/MVDVAE/TCGA_toydata/'
with open(path + 'TCGA_mRNAs.csv') as f:
    geneids = f.readline().strip().split(',')
print(len(geneids))

# Load the 2 input datasets
X1 = np.loadtxt(path + 'TCGA_mRNAs2.csv', delimiter=",", skiprows=1)
X2 = np.loadtxt(path + 'TCGA_DNAm.csv', delimiter=",", skiprows=1)
print(X1.shape)
print(X2.shape)

# Select the most variable features
X1_var = np.argsort(np.var(X1, axis=0))[::-1]
X1 = X1[:, X1_var[:2000]]
X2_var = np.argsort(np.var(X2, axis=0))[::-1]
X2 = X2[:, X2_var[:2000]]

# Normalize datasets
X1 = MinMaxScaler().fit_transform(X1)
X2 = MinMaxScaler().fit_transform(X2)
print(np.quantile(X1, [0,0.1,0.5,0.9,1]))
print(np.quantile(X2, [0,0.1,0.5,0.9,1]))

# Save pre-processed datasets
#np.savetxt(path + 'TCGA_mRNA2_processed.csv', X1, delimiter=",")
#np.savetxt(path + 'TCGA_DNAm_processed.csv', X2, delimiter=",")

def generate_conf_linear(X1, X2):
    # Apply linear artificial confounder
    conf = np.random.choice(6, size=X1.shape[0])
    # confounding effect is linear to the confounder
    conf2 = conf + 5
    # the confounder has different weights on different features
    Ws_X1 = np.random.uniform(0, 0.1, X1.shape[1])
    Ws_X2 = np.random.uniform(0, 0.2, X2.shape[1])
    # add the weighted confounding effect to original features
    print(np.quantile(X1, [0, 0.1, 0.5, 0.9, 1]))
    X1_conf = X1 + np.outer(conf2, Ws_X1)
    print(np.quantile(X1_conf, [0, 0.1, 0.5, 0.9, 1]))
    print(np.quantile(X2, [0, 0.1, 0.5, 0.9, 1]))
    X2_conf = X2 + np.outer(conf2, Ws_X2)
    print(np.quantile(X2_conf, [0, 0.1, 0.5, 0.9, 1]))

    return X1_conf, X2_conf, conf


def generate_conf_square(X1, X2):
    # Apply square artificial confounder
    conf = np.random.choice(6, size=X1.shape[0])
    # confounding effect is square to the confounder
    conf2 = np.square(conf)
    # the confounder has different weights on different features
    Ws_X1 = np.random.uniform(0, 0.04, X1.shape[1])
    Ws_X2 = np.random.uniform(0, 0.04, X2.shape[1])
    # add the weighted confounding effect to original features
    print(np.quantile(X1, [0, 0.1, 0.5, 0.9, 1]))
    X1_conf = X1 + np.outer(conf2, Ws_X1)
    print(np.quantile(X1_conf, [0, 0.1, 0.5, 0.9, 1]))
    print(np.quantile(X2, [0, 0.1, 0.5, 0.9, 1]))
    X2_conf = X2 + np.outer(conf2, Ws_X2)
    print(np.quantile(X2_conf, [0, 0.1, 0.5, 0.9, 1]))

    return X1_conf, X2_conf, conf


def generate_conf_categ(X1, X2):
    n_classes = 6
    # Apply categorical artificial confounder
    conf = np.random.choice(n_classes, size=X1.shape[0])
    # confounding effect is categorical to the confounder
    conf_X1 = np.random.uniform(0, 1, [n_classes, X1.shape[1]])[conf]
    conf_X2 = np.random.uniform(0, 1, [n_classes, X2.shape[1]])[conf]
    # the confounder has different weights on different samples
    #Ws = np.random.uniform(0.3, 1, size=(X1.shape[0],1)) # categ1
    Ws = np.random.uniform(0, 1, size=(X1.shape[0],1)) # categ2
    # add the weighted confounding effect to original features
    print(np.quantile(X1, [0, 0.1, 0.5, 0.9, 1]))
    X1_conf = X1 + conf_X1 * Ws
    print(np.quantile(X1_conf, [0, 0.1, 0.5, 0.9, 1]))
    print(np.quantile(X2, [0, 0.1, 0.5, 0.9, 1]))
    X2_conf = X2 + conf_X2 * Ws
    print(np.quantile(X2_conf, [0, 0.1, 0.5, 0.9, 1]))

    return X1_conf, X2_conf, conf


X1_conf, X2_conf, conf1 = generate_conf_linear(X1, X2)
X1_conf, X2_conf, conf2 = generate_conf_square(X1_conf, X2_conf)
X1_conf, X2_conf, conf3 = generate_conf_categ(X1_conf, X2_conf)
# Save confounded datasets
#np.savetxt(path + 'TCGA_confounder_categ2.csv', conf[:,None])
np.savetxt(path + 'TCGA_confounder_multi.csv', np.column_stack((conf1, conf2, conf3)), delimiter=",", fmt="%d")
np.savetxt(path + 'TCGA_mRNA2_confounded_multi.csv', X1_conf, delimiter=",")
np.savetxt(path + 'TCGA_DNAm_confounded_multi.csv', X2_conf, delimiter=",")



