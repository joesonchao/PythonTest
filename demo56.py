import tensorflow as tf
import keras
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)

trainImages /= 255
testImages /= 255

NUM_DIGITS = 10
trainLabels = keras.utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = keras.utils.to_categorical(test_labels, NUM_DIGITS)
print("data prepare ready..")

model1 = keras.Sequential()
model1.add(keras.layers.Dense(units=256, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
# model1.add(keras.layers.Dense(units=64, activation=tf.nn.relu))
model1.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model1.summary())
# make a tensorboard callback
tb1 = keras.callbacks.TensorBoard(log_dir='c:\\logs', histogram_freq=0, write_graph=True,
                                  write_images=True)
model1.fit(trainImages, trainLabels, epochs=2000, batch_size=200, verbose=0, callbacks=[tb1])

loss, accuracy = model1.evaluate(testImages, testLabels)
p1 = model1.predict(testImages)
p2 = model1.predict_classes(testImages)
p3 = model1.predict_proba(testImages)
print(p1[:10])
print(p2[:10])
print(p3[:10])
print('accuracy = {}'.format(accuracy))