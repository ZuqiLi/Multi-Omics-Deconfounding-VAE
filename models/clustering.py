from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score


def kmeans(X, c):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clust = KMeans(n_clusters=c, random_state=0, n_init=10).fit(X)
    return clust.labels_


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




