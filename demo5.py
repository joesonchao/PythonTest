from sklearn import linear_model

features = [[0, 0], [1, 1], [2, 2]]
values = [1, 4, 8]

regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print('coefficient={}'.format(regression1.coef_))
print('intercpt={}'.format(regression1.intercept_))

print(regression1.predict([[0.8, 0.8], [2, 1], [4, 4], [4, 6]]))

print('r square={}'.format(regression1.score(features,[1,4,10])))
