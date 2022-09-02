import tensorflow as tf
import numpy as np

v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(1))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("######print get_collection")
    print(tf.get_collection('loss'))
    print("######print run get_collection")
    print(sess.run(tf.get_collection('loss')))
    print("######print type run get_collection")
    print(type(sess.run(tf.get_collection('loss'))))
    print("######print add_n run get_collection")
    print(sess.run(tf.add_n(tf.get_collection('loss'))))
