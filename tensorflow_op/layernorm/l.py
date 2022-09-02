import tensorflow as tf

x1 = tf.convert_to_tensor(
    [[[18.369314, 2.6570225, 20.402943],
      [10.403599, 2.7813416, 20.794857]],
     [[19.0327, 2.6398268, 6.3894367],
      [3.921237, 10.761424, 2.7887821]],
     [[11.466338, 20.210938, 8.242946],
      [22.77081, 11.555874, 11.183836]],
     [[8.976935, 10.204252, 11.20231],
      [-7.356888, 6.2725096, 1.1952505]]])
mean_x = tf.reduce_mean(x1, axis=-1)
mean_xx = tf.reduce_mean(x1, axis=0)
mean_xxx = tf.reduce_mean(x1, axis=1)
print(mean_x.shape)  # (4, 2)
mean_x = tf.expand_dims(mean_x, -1)

std_x = tf.math.reduce_std(x1, axis=-1)
print(std_x.shape)  # (4, 2)
std_x = tf.expand_dims(std_x, -1)

# 手动计算
la_no1 = (x1-mean_x)/std_x

x = tf.placeholder(tf.float32, shape=[4, 2, 3])
la_no = tf.contrib.layers.layer_norm(
      inputs=x, begin_norm_axis=1, begin_params_axis=-1)
with tf.Session() as sess1:
    sess1.run(tf.global_variables_initializer())
    x1 = sess1.run(x1)
    print(sess1.run(mean_xxx))
    print("    ----")
    print(sess1.run(la_no1))
    print("    ----")
    print(sess1.run(la_no, feed_dict={x: x1}))

