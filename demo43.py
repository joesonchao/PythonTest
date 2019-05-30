import numpy as np

dataset1 = np.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

#inputList = dataset1[:, 0:8]
#resultList = dataset1[:, 8]
inputList = dataset1[:, 0:-1]
resultList = dataset1[:, -1]
print("input shape={}".format(inputList.shape))
print("result shape={}".format(resultList.shape))
