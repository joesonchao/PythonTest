import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

regressionData1 = datasets.make_regression(100, 1, noise=5)
#noise會修改畫出來的數值
print(type(regressionData1))
print(regressionData1[0].shape, regressionData1[1].shape)
regression1 = linear_model.LinearRegression()
regression1.fit(regressionData1[0], regressionData1[1])
print("coef={}, intercept={}".format(regression1.coef_, regression1.intercept_))
print('score={}'.format(regression1.score(regressionData1[0], regressionData1[1])))
range1 = [-3, 3]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_)
plt.scatter(regressionData1[0], regressionData1[1],c='r')
plt.show()