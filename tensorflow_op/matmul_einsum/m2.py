import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)


E = 2
G = 1
C = 2
M = 3

H = 2


x1 = np.full((E, G, C, M), 1.0)
x2 = np.full((E,G,C,M), 2.0)
x = np.vstack([x1, x2])

y1 = np.full((E, M, H), 3.0)
y2 = np.full((E, M, H), 2.0)
y = np.vstack([y1, y2])

x = tf.convert_to_tensor(x)
print("x shape ", x.shape)
y = tf.convert_to_tensor(y)

x_v = tf.gather(x, [0,1], axis=0)
y_v = tf.gather(y, [0,1], axis=0)


e = tf.einsum('EGCM,EMH->EGCH',x,y)
e_v = tf.einsum('EGCM,EMH->EGCH',x_v,y_v)

indices = tf.constant([[0], [1]])
shape = tf.constant([4,1,2,2])

e_e = tf.scatter_nd(indices, e_v, shape)

with tf.Session() as sess:
    print(sess.run(x))
    print("######## x_valid: ")
    x_ = sess.run(x_v)
    print("x_ shape ", x_.shape)
    print("######## e")
    print(sess.run(e))
    print("######## e_v")
    print(sess.run(e_v))
    print("######## e_e")
    print(sess.run(e_e))
    
