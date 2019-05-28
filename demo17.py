import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()

pca = PCA(n_components=2)

data = pca.fit(iris.data).transform(iris.data)

print(data.shape)
print(iris.data.shape)

datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
print(datamax)
print(datamin)
n = 2000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
#kernel='linear'|rbf|poly (x)sigmoid
#C=1 100
svc = svm.SVC(kernel='rbf', C=100)
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])

plt.contour(X, Y, Z.reshape(X.shape), colors='K')

for c, s in zip([0, 1, 2], ['o', '+', 'x']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
plt.show()