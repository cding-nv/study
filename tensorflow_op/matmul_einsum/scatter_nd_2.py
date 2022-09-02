import tensorflow as tf

indices = tf.constant([[0,0,0], [0,1,2]])
updates = tf.constant([[5, 5, 5, 5], [6, 6, 6, 6]])
shape = tf.constant([2, 4, 4, 4])
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
  print(sess.run(scatter))
