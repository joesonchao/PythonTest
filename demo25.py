import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]

print(X.shape)
K = 5

kmeans = KMeans(n_clusters=K)
kmeans.fit(X)

colors = ['c', 'm', 'y', 'k', 'b']
markers = ['o', 'v', '*', 'x', '^']

for i in range(K):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
    print(dataX.shape)
plt.show()
