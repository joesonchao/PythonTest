from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

print(list(iris.keys()))
X = iris['data'][:, 3:]
y = (iris["target"] == 2).astype(np.int)

regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_, regression1.intercept_)

X_sequence = np.linspace(0, 3.5, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(X_sequence)

plt.plot(X, y, 'gs')
plt.plot(X_sequence, y_prob[:,1],'b--', label='iris-virginica')
plt.plot(X_sequence, y_prob[:,0],'r-', label='Non iris-virginica')
plt.xlabel('petal width', fontsize=14)
plt.ylabel('prob',fontsize=14)
plt.legend(fontsize=14)
plt.show()