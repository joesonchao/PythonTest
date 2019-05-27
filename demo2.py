import matplotlib.pyplot as plt
import numpy as np

rangel = [-1, 3]
p = np.array([3])
print((type(rangel), type(p)))
plt.plot(rangel, p * rangel - 5, c='green')
plt.show()
