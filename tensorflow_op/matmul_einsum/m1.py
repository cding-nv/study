import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)


E = 2
G = 1
C = 2
M = 3

H = 2


x = np.full((E, G, C, M), 1.0)
y = np.full((E, M, H), 2.0)

x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

z = tf.matmul(x,y)

e = tf.einsum('EGCM,EMH->EGCH',x,y)

with tf.Session() as sess:
    print(sess.run(z))
    print("########")
    print(sess.run(e))
