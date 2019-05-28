from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [1, 6], [4, 2], [4, 4], [4, 0], [4, 6], [4, 7]])
kmeans = KMeans(n_clusters=2).fit(X)
print("label=", kmeans.labels_)
print("predict:", kmeans.predict([[0, 0], [6, 6], [-3, -3]]))
print("centers:", kmeans.cluster_centers_)
