from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

X = np.r_[np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
print(X)
[plt.scatter(e[0], e[1], c='g', s=7) for e in X]
# find k (kmean, center point)
k = 3
C_x = np.random.randint(np.min(X), np.max(X), size=k)
C_y = np.random.randint(np.min(X), np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
plt.scatter(C_x, C_y, marker='*', c='#C02244')
plt.show()


def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


C_old = np.zeros(C.shape)
delta = dist(C, C_old, None)
clusters = np.zeros((len(X)))


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if current_cluster[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.scatter(C[:, 0], C[:, 1], marker='*', c='#C02244')
    plt.title('delta will be:%.4f' % delta)
    plt.plot()
    plt.show()


while delta != 0:
    print('start a new interation')
    # calculate each point distances and assign to new cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    # move kmeans center to new place
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    # calculate distance from current clusters
    delta = dist(C, C_old, None)
    plot_kmean(clusters, delta)
