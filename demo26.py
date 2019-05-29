import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(5000, 2) + [2, 2],
          np.random.randn(5000, 2) + [0, -2],
          np.random.randn(5000, 2) + [-2, 2]]

# print(X.shape)
K = 10

kmeans = KMeans(n_clusters=K)
kmeans.fit(X)

colors = ['c', 'm', 'y', 'k', 'b']
markers = ['o', 'v', '*', 'x', '^']
print('kmean center:', kmeans.cluster_centers_)
print('label:', kmeans.labels_)
print('iteration:', kmeans.n_iter_)
print('cost:', kmeans.inertia_)
