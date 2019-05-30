import numpy
from keras.datasets import imdb
from matplotlib import pyplot

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X = numpy.concatenate((X_train, X_test), axis=0)
print(X.shape)
y = numpy.concatenate((y_train, y_test), axis=0)
print(y.shape)
print(X[0])
print(y[:100])
print(numpy.unique(y))
print(len(numpy.unique(numpy.hstack(X))))
print("every review length")
result = [len(x) for x in X]
print("mean length: %.3f, std:%.3f" % (numpy.mean(result), numpy.std(result)))

pyplot.subplot(1, 2, 1)
pyplot.boxplot(result)
pyplot.subplot(1, 2, 2)
pyplot.hist(result)
pyplot.show()