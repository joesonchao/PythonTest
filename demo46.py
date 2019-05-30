import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

dataset1 = np.loadtxt("data\\diabetes.csv", delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)


inputList = dataset1[:, 0:-1]
resultList = dataset1[:, -1]

# generate k-fold
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []
print(type(fiveFold))


print("input shape={}".format(inputList.shape))
print("result shape={}".format(resultList.shape))
print(np.unique(resultList))  # 0/1 only, classification
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

for train,test in fiveFold.split(inputList, resultList):

    model.fit(inputList[train], resultList[train], epochs=200, batch_size=29,verbose=0)

    scores = model.evaluate(inputList[test], resultList[test],verbose=0)
    totalScores.append(scores[1]*100)
    print("get a result score=%.3f"%(scores[1]*100))
print("total 5 result mean:%.3f, std:%.3f\n"%(np.mean(totalScores),np.std(totalScores)))