import tensorflow as tf
import numpy as np

t2 = np.zeros((2, 3))
t3 = tf.expand_dims(t2, 0)
t4 = tf.expand_dims(t2, 1)

with tf.Session() as sess:
    print(t2)
    print("====")
    print(sess.run(t3))
    print("====")
    print(sess.run(t4))
