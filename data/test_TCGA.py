import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score, normalized_mutual_info_score


# Set seeds for replicability
np.random.seed(1234)
torch.manual_seed(1234)

# Set PATHs
PATH_data = "Data"

# Load data
X1 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_mRNAs_processed.csv'), delimiter=",")
X2 = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_miRNAs_processed.csv'), delimiter=",")
X1 = torch.from_numpy(X1).to(torch.float32)
X2 = torch.from_numpy(X2).to(torch.float32)
#print(X1.shape, X2.shape)
traits = np.loadtxt(os.path.join(PATH_data, "TCGA",'TCGA_clinic.csv'), delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
traits = torch.from_numpy(traits).to(torch.float32)
# Plot the correlations between covariates and cancers
fig, ax = plt.subplots()
im = plt.imshow(np.corrcoef(traits.T), cmap='hot', interpolation='nearest')
ax.set_xticks(np.arange(traits.shape[1]), labels=['Stage','Age','Race','Gender','Cancer'])
ax.set_yticks(np.arange(traits.shape[1]), labels=['Stage','Age','Race','Gender','Cancer'])
ax.tick_params(axis='both', labelsize=20)
plt.colorbar(im)
plt.show()
# Cancer types as phenotype
Y = traits[:, -1]
Y_m = torch.nn.functional.one_hot(Y.long()-1).to(torch.float32)
# The rest as confounders
conf = traits[:, :-1]

# Remove confounding effects from every feature
for i in range(conf.shape[1]):
    lm = LinearRegression().fit(Y_m, conf[:,[i]])
    conf[:,[i]] -= lm.predict(Y_m)
for i in range(X1.shape[1]):
    lm = LinearRegression().fit(conf, X1[:,[i]])
    X1[:,[i]] -= lm.predict(conf)
for i in range(X2.shape[1]):
    lm = LinearRegression().fit(conf, X2[:,[i]])
    X2[:,[i]] -= lm.predict(conf)

# Test on the whole dataset for clustering
n_samples = X1.shape[0]
indices = np.random.permutation(n_samples)
test_idx = indices
X1_test, X2_test = X1[test_idx,:], X2[test_idx,:]
Y_test = Y[test_idx]

# Run k-means clustering and compute the metrics
X = StandardScaler().fit_transform(np.concatenate((X1_test, X2_test), 1))
clust = KMeans(n_clusters=6, random_state=0, n_init=10).fit(X).labels_
print(silhouette_score(X, clust))
print(davies_bouldin_score(X, clust))
print(adjusted_rand_score(Y_test, clust))
print(normalized_mutual_info_score(Y_test, clust))

