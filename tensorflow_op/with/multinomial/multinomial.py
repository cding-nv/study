import tensorflow as tf
#samples = tf.multinomial(tf.log([[10.,10.,10.]]), 5)
a = tf.log([[10., 10., 10.]])
samples = tf.multinomial(a, 5)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(samples))
