import numpy as np
from keras.layers import Dense
from keras.models import Sequential

dataset1 = np.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

# inputList = dataset1[:, 0:8]
# resultList = dataset1[:, 8]
inputList = dataset1[:, 0:-1]
resultList = dataset1[:, -1]
print("input shape={}".format(inputList.shape))
print("result shape={}".format(resultList.shape))

model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(inputList, resultList, epochs=200, batch_size=20)

scores = model.evaluate(inputList, resultList)
print("score=", scores)
print(type(model.metrics_names))
print("matrics=", model.metrics_names)
print("%s:%.3f\n" % (model.metrics_names[0], scores[0]))
print("%s:%.3f\n" % (model.metrics_names[1], scores[1]))
