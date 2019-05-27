import matplotlib.pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
features = [[1], [2], [3]]
values = [1, 4, 9]
plt.scatter(features, values, c='green')
plt.show()
regression1.fit(feature, values)
# y=ax+b, a==> coef, b==>intercept
print('coefficient={}'.format(regression1.coef_))
print('intercept={}'.format(regression1.intercept_))
range1 = [0, 3]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='gray')
print('score='c, regression1.score(features, values))
plt.show()
