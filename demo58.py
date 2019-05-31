import numpy as np
from keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()
print(train_data.shape, test_data.shape)
print(train_target[:10])
print(test_target[:10])

mean = train_data.mean(axis=0)
print(mean)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

print('data prepare ready')
from keras import models, layers


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit(train_data, train_target, validation_split=0.1,
          epochs=100, batch_size=10, verbose=1)