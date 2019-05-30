import tensorflow as tf

v1 = [-2.0, -1.0, -0.005, 0.5, 2, 5]

result1 = tf.nn.relu(v1)
result2 = tf.nn.sigmoid(v1)

with tf.Session() as session1:
    print("relu result:", session1.run(result1))
    print("sigmoid result:", session1.run(result2))
