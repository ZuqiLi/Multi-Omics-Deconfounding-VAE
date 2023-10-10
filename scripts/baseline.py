import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score, normalized_mutual_info_score
from scipy import stats


# Set seeds for replicability
np.random.seed(1234)
torch.manual_seed(1234)

# Set PATHs
PATH_data = "./"

# Load data
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA_toydata",'TCGA_mRNA2_confounded_multi.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA_toydata",'TCGA_DNAm_confounded_multi.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
X2 = torch.from_numpy(X2).to(torch.float32)
print(X1.shape, X2.shape)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA_toydata",'TCGA_clinic2.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
traits = torch.from_numpy(traits).to(torch.float32)
print(traits.shape)
# Cancer types as phenotype
Y = traits[:, -1]
n_samples = X1.shape[0]

# load the artificial confounder
conf = np.loadtxt(os.path.join(PATH_data, "TCGA_toydata",'TCGA_confounder_multi.csv'), delimiter=',')
conf = conf[:,:].astype(np.float32)
# one-hot encode the categorical confounder
conf_categ = OneHotEncoder(sparse=False, dtype=np.float32).fit_transform(conf[:,-1:])
conf = np.concatenate((conf[:,:-1], conf_categ), axis=1)
print(conf.shape)

# Remove confounding effects from every feature
#for i in range(X1.shape[1]):
#    lm = LinearRegression().fit(conf, X1[:,[i]])
#    X1[:,[i]] -= lm.predict(conf)
#for i in range(X2.shape[1]):
#    lm = LinearRegression().fit(conf, X2[:,[i]])
#    X2[:,[i]] -= lm.predict(conf)

# Test on the whole dataset for clustering
indices = np.random.permutation(n_samples)
test_idx = indices
X1_test, X2_test = X1[test_idx,:], X2[test_idx,:]
Y_test = Y[test_idx]
conf_test = conf[test_idx, :]

# Run k-means clustering and compute the metrics
#X = np.concatenate((X1_test, X2_test), 1)
X = X1_test
X = StandardScaler().fit_transform(X)
# PCA to reduce dimensionality
#X = PCA(n_components=50).fit_transform(X)
clust = KMeans(n_clusters=6, random_state=0, n_init=10).fit(X).labels_
print(silhouette_score(X, clust))
print(davies_bouldin_score(X, clust))
print("====================")

# Metrics on true clusters
print(adjusted_rand_score(Y_test, clust))
print(normalized_mutual_info_score(Y_test, clust))
print("====================")

# Metrics on confounder
conf = conf_test[:,0]
print(adjusted_rand_score(conf, clust))
print(normalized_mutual_info_score(conf, clust))
conf = conf_test[:,1]
print(adjusted_rand_score(conf, clust))
print(normalized_mutual_info_score(conf, clust))
conf = np.argmax(conf_test[:,2:], 1)
print(adjusted_rand_score(conf, clust))
print(normalized_mutual_info_score(conf, clust))
print("====================")

# Kruskal-Wallis ANOVA
labels = [conf[clust == label] for label in np.unique(clust)]
pval = stats.kruskal(*labels)[1]
print(pval)
tab = stats.contingency.crosstab(conf, clust)
res = stats.contingency.chi2_contingency(tab[1])
print(res[1])

