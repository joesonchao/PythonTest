import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 3, 2, 2, 3])
classifier1 = SVC()
classifier1.fit(X, y)
print("predict=", classifier1.predict([[-0.8, -1], [4, 4], [3, -3], [-3, 3]]))