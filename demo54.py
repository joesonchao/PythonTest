import matplotlib.pyplot as plt
import keras.datasets as datasets

(train_images, train_labels), (test_images, test_labels) = \
    datasets.mnist.load_data()
print("train shape=%s, test shape=%s" % (train_images.shape, test_images.shape))
print("train label length=%d, test label length=%d" % (len(train_labels), len(test_labels)))


def plotImage(index):
    plt.title("train image marked as %d" % train_labels[index])
    plt.imshow(train_images[index], cmap="binary")
    plt.show()
    pass


def plotTestImage(index):
    plt.title("test image marked as %d" % test_labels[index])
    plt.imshow(test_images[index], cmap='binary')
    plt.show()
    pass


plotImage(500)
plotTestImage(1000)
