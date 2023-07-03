import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection as FDR
from scipy.stats import chi2_contingency


def kmeans(X, c):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clust = KMeans(n_clusters=c, random_state=0, n_init=10).fit(X)
    return clust.labels_


def consensus_clustering(labels, c):
    # compute co-occurrence matrix
    n_samples = len(labels[0])
    mat = np.zeros((n_samples, n_samples), np.int32)
    for label in labels:
        for i, li in enumerate(label):
            for j, lj in enumerate(label):
                if li == lj: mat[i,j] += 1
    mat = mat / len(labels)

    # compute the dispersion score
    disp = np.sum(4 * np.square(mat - 0.5)) / mat.shape[0] / mat.shape[1]

    # spectral clustering
    sc = SpectralClustering(c, n_init=10, affinity='precomputed', assign_labels='kmeans').fit(mat)
    label = sc.labels_

    return label, mat, disp


def internal_metrics(X, labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    SS = silhouette_score(X, labels, metric='euclidean')
    DB = davies_bouldin_score(X, labels)    
    return SS, DB


def external_metrics(clust_labels, true_labels):
    ARI = adjusted_rand_score(true_labels, clust_labels)
    NMI = normalized_mutual_info_score(true_labels, clust_labels)
    return ARI, NMI


def test_confounding(clust, conf):
    pvals = []
    for var in conf.T:
        #if np.all([x.is_integer() for x in var]): # integer vector (discrete variable)
        #    # Chi-square test
        #    count = pd.crosstab(index=clust, columns=var)
        #    chi = chi2_contingency(count)
        #    pvals.append(chi[1])
        #else: # continuous variable
        # ANOVA
        aov = ols('cov ~ C(clust)', data={'cov': var, 'clust': clust}).fit()
        aov = sm.stats.anova_lm(aov, typ=2)
        pvals.append(aov['PR(>F)'][0])
    return pvals


def test_embedding_confounding(LF, conf):
    # Linear model F-test: every covariate ~ all G1 vectors
    LF = sm.add_constant(LF) # An intercept is not added by default so it has to be added manually
    pvals = [] # p-value of the F-statistic
    arsqs = [] # adjusted R-squared
    for i in range(conf.shape[1]):
        lm = sm.OLS(conf[:,i], LF).fit()
        pvals.append(lm.f_pvalue)
        arsqs.append(lm.rsquared_adj)
    # multiple testing correction
    pvals = FDR(pvals)[1]
    return pvals, arsqs



