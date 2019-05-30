import numpy as np
from keras.layers import Dense
from keras.models import Sequential

dataset1 = np.loadtxt("data\\diabetes.csv", delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

# inputList = dataset1[:, 0:8]
# resultList = dataset1[:, 8]
inputList = dataset1[:, 0:-1]
resultList = dataset1[:, -1]
print("input shape={}".format(inputList.shape))
print("result shape={}".format(resultList.shape))
print(np.unique(resultList))  # 0/1 only, classification
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# validation_split
history = model.fit(inputList, resultList, epochs=200, batch_size=20, validation_split=0.1)
print(type(history.history))