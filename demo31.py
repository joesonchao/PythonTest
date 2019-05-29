import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
#Y = np.array([1, 2, 2, 1, 2, 2])
Y = np.array([1, 1, 2, 1, 1, 2])
#Y = np.array([1, 3, 2, 1, 3, 2])
x_min, x_max = -4, 4
y_min, y_max = -4, 4
h = .025
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
clf1 = GaussianNB()
clf1.fit(X, Y)
Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z)
# plt.show()

XB = []
YB = []
XR = []
YR = []
index = 0
for index in range(0, len(Y)):
    if Y[index] == 1:
        print("B equal to", X[index, :])
        XB.append(X[index, 0])
        YB.append(X[index, 1])
    elif Y[index] == 2:
        print("R equal to", X[index, :])
        XR.append(X[index, 0])
        YR.append(X[index, 1])
print(X)
print(X[:, 0])
plt.scatter(XB, YB, color='b', label="Blue")
plt.scatter(XR, YR, color='r', label="Red")
plt.legend()
plt.show()