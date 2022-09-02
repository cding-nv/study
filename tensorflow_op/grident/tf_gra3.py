import tensorflow as tf

w1 = tf.get_variable('w1', shape=[2])
w2 = tf.get_variable('w2', shape=[2])

w3 = tf.get_variable('w3', shape=[2])
w4 = tf.get_variable('w4', shape=[2])

z1 = w1 + w2+ w3
z2 =  w3 + w4

grads = tf.gradients([z1, z2], [w1, w2, w3, w4],grad_ys=[[1, 2], [1, 2]])
#grads = tf.gradients([z1, z2], [w1, w2, w3, w4])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(grads)) 
