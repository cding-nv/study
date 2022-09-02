import tensorflow as tf
import numpy as np

#input = tf.constant([1,1,1,2,2,2,1,1,1,2,2,2,1,1,1,2,2,2,1,1,1,2,2,2,1,1,1,2,2,2,1,1,1,2,2,2], dtype=tf.float32, shape=(2,2,3,3))

np_input1 = np.full((1, 2, 3, 3), 1.0)
np_input2 = np.full((1, 2, 3, 3), 2.0)
np_input = np.vstack([np_input1, np_input2])
input = np_input.astype(np.float32)

# column
op0 = tf.nn.softmax(input, axis=0) 

# row
op1 = tf.nn.softmax(input, axis=1) 

op2 = tf.nn.softmax(input, axis=-1)

i = tf.identity(input)
with tf.Session() as sess:
    print(input)
    print("Peak")
    print(sess.run(op0))
    print("Peak")
    print(sess.run(op1))
    print("Peak")
    print(sess.run(op2))
