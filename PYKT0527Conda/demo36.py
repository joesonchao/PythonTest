import tensorflow as tf

hello1 = tf.constant("hello tensorflow from pycharm")
session = tf.Session()
print(session.run(hello1))
session.close()

with tf.Session() as session2:
    print(session2.run(hello1))