import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from scipy import sparse, linalg


def find_eigen_v(L, num_eigenv):
    eigenvals, eigenvects = linalg.eig(L)
    eigenvals = np.real(eigenvals)
    eigenvects = np.real(eigenvects)
    eigenvals_sorted_idx = np.argsort(eigenvals)
    indices = eigenvals_sorted_idx[: num_eigenv]
    eigenvect_df = pd.DataFrame(eigenvects[:, indices.squeeze()])
    eigenvect_df.columns = ['v_' + str(c) for c in eigenvect_df.columns]

    return eigenvect_df


def create_lap_mat(df, nn):
    # Adjacency Matrix.
    adj_m = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')
    adj_m_w = (1 / 2) * (adj_m + adj_m.T)
    # Graph Laplacian.
    L = sparse.csgraph.laplacian(csgraph=adj_m_w, normed=False)
    L = L.toarray()
    return L


def run_spec_clustering(df, num_clusters):
    sigma = 500.0
    sc = SpectralClustering(num_clusters, gamma=1.0 / sigma ** 2.0, affinity='rbf', n_init=100, random_state=0,
                            assign_labels='kmeans').fit(df)
    return sc.labels_


# Spectral clustering
def spectral_clustering(df, neighbors, num_eigenv, num_clusters):
    L = create_lap_mat(df, neighbors)
    new_df = find_eigen_v(L, num_eigenv)
    cluster = run_spec_clustering(new_df, num_clusters)
    return cluster


df = pd.read_csv('./static/im_features.csv')
cluster = spectral_clustering(df=df[df.columns[0:4096]], neighbors=10, num_eigenv=30, num_clusters=10)
df['cluster'] = pd.Series(cluster, index=df.index)
centroids = df.groupby(["cluster"]).mean()

# save data
df.to_csv("./static/clusters.csv", index=False)
centroids.to_csv("./static/centroids.csv", index=False)
