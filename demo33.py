from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
