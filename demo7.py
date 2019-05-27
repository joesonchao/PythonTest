import matplotlib.pyplot as plt
from sklearn import datasets

regressionData = datasets.make_regression(10, 6, noise=10)
for i in range(0, 6):
    x1 = regressionData[0][:, 0]
    y = regressionData[1]
    plt.scatter(x1, y)
    # plt.show()

regressionX = regressionData[0]
print(type(regressionX))

x1 = sorted(regressionX, key=lambda tup: tup[0])
print(x1)
x2 = sorted(regressionX, key=lambda tup: tup[1])
print(x2)
x6 = sorted(regressionX, key=lambda tup: tup[5])
print(x6)
