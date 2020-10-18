
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


def RRE(X, X_hat):
    rre = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    
    return rre


def MA(R, Y):
    Y_pred = assign_cluster_label(R, Y)
    acc = accuracy_score(Y, Y_pred)
    return acc

def assign_cluster_label(X, Y):
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred
