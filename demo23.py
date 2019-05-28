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
