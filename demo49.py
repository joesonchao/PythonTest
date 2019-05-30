from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

dataFrame1 = read_csv('data\\iris.data', header=None)
print(type(dataFrame1), dataFrame1)
dataset = dataFrame1.values
print(type(dataset))
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(type(features), type(labels))

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(type(encoded_Y), encoded_Y[:10], encoded_Y[50:60], encoded_Y[100:110])
dummy_y = np_utils.to_categorical(encoded_Y)
print(type(dummy_y), dummy_y[:3], dummy_y[50:53], dummy_y[100:103])


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


estimator1 = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)
kfold1 = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator1, features, dummy_y, cv=kfold1)
print("acc:%.4f, std:%.4f"%(results.mean(), results.std()))