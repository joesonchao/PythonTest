import numpy as np
from sklearn import linear_model, datasets

diabestes = datasets.load_diabetes()
print(type(diabestes))
print(diabestes.data.shape)
print(diabestes.target.shape)

dataForTest = -50
data_train = diabestes.data[:dataForTest]
target_train = diabestes.target[:dataForTest]
print(data_train.shape, target_train.shape)
data_test = diabestes.data[dataForTest:]
target_test = diabestes.target[dataForTest:]

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print("regression finish")
print('score:', regression1.score(data_test, target_test))
