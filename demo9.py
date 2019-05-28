import numpy as np

a = np.zeros((10, 2))
print(a)
b = a.T
print(b)
c = b.view()
print(c)
print(a.shape, b.shape, c.shape)
d = np.reshape(b, (5, 4))
print(d)
e = np.reshape(b, (20,))  # 1d array
print(e)
f = np.reshape(b, (20, -1))  # 2d array
print(f)
g = np.reshape(b, (20, 1))  # 2d array
print(g)
h = np.reshape(b, (-1, 20))
print(h)
print(d.shape, e.shape, f.shape, g.shape, h.shape)