import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0,1],[1,3],[2,8]]
values = [1,4,5.5]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print('coefficient={}'.format(regression1.coef_))
print('intercept={}'.format(regression1.intercept_))
print('first element={}, second element={}'.format(regression1.coef_[0],regression1.coef_[1]))
plt.scatter([[0],[1],[2]],[1,4,5],c='green')
plt.scatter([[1],[3],[8]],[1,5,5.5],c='blue')
plt.show()
