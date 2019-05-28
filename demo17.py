import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()

pca = PCA(n_components=2)

data = pca.fit(iris.data).transform(iris.data)

print(data.shape)
print(iris.data.shape)