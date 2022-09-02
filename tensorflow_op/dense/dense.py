import tensorflow as tf

inputs = tf.ones([2,20])
a = tf.layers.dense(inputs, 60)
print(inputs.get_shape())
print(a.get_shape())
