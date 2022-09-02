import tensorflow as tf

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1) # 对x进行加1，x_plus_l是个op
with tf.control_dependencies([x_plus_1]):
    y = x
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run() # 相当于session.run(init)
    for i in range(5):
        sess.run(x_plus_1)
        print(y.eval()) # y.eval()这个相当于session.run(y)
