scores = [4.0, 1.0, 2.0, 1.0, 1.0]
import numpy as np
import tensorflow as tf

def manualSoftMax(x):
    y = np.array(x)
    return np.exp(y) / np.sum(np.exp(y), axis=0)

print(manualSoftMax(scores))

result1 = tf.nn.softmax(scores)
with tf.Session() as session1:
    print(session1.run(result1))