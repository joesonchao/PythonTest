from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(list(iris.keys()))

x = iris['data'][:,3:]
y = (iris["target"]==2).astype(np.int)

regression1 = LogisticRegression()
regression1.fit(x,y)
print(regression1.coef_, regression1.intercept_)
