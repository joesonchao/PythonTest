import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = neighbors.kneighbors(X, return_distance=True)
# distance [self, nearest neighbors]
print(distances)
# indices from distance [self, neighbor id]
print(indices)

print(neighbors.kneighbors_graph(X).toarray())
