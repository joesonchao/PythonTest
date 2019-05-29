import tensorflow as tf

a = tf.placeholder(dtype=tf.int32, shape=(None,))
b = tf.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
print(c)
with tf.Session() as session1:
    result = session1.run(c, feed_dict={
        a: [3, 4, 5],
        b: [5, 6, 7]
    })
    print(result)
