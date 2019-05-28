import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes))
print(diabetes.data.shape)
print(diabetes.target.shape)
dataForTest = -50
data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print(data_train.shape, target_train.shape)
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print("regression finished")
print('score:',regression1.score(data_test, target_test))

for i in range(dataForTest,0):
    dataArray = np.array(data_test[i]).reshape(1,-1)
    print('predict/actual:',regression1.predict(dataArray)[0], target_test[i])