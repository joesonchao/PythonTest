from keras.datasets import reuters
import numpy

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data), len(train_labels))
print(numpy.unique(train_labels))

word_index = reuters.get_word_index()
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
decodedFirstNews = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decodedFirstNews)
decoded10thNews = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[9]])
print(decoded10thNews)

import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
print(one_hot_train_labels[:10])
print(one_hot_test_labels[:10])

from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, one_hot_train_labels, epochs=20, batch_size=20,
                    validation_data=(x_test, one_hot_test_labels))
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b.-', label='Training loss')
plt.plot(epochs, val_loss, 'ro--', label='validate loss')
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, 'b.-', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro--', label='validate accuracy')
plt.show()