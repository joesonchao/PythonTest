import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score

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


def create_default_model():
    model = Sequential()
    model.add(Dense(64, input_dim=8, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


from keras.wrappers.scikit_learn import KerasClassifier

model1 = KerasClassifier(build_fn=create_default_model, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model1, inputList, resultList, cv=fiveFold)
print("mean=%.3f, std=%.3f" % (results.mean(), results.std()))