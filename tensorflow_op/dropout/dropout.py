import tensorflow as tf

drop = tf.placeholder(tf.float32)
x = tf.Variable(tf.ones([5,5]))
y = tf.nn.dropout(x, drop)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    init.run()
    print(sess.run(y, feed_dict = {drop:0.2}))
