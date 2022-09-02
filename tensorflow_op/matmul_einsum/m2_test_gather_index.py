import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)


E = 2
G = 1
C = 4
M = 3

H = 2


x1 = np.full((E, G, C, M), 1.0)
x2 = np.full((E,G,C,M), 0.0)
x = np.vstack([x1, x2])

#y1 = np.full((E, M, H), 2.0)
#y2 = np.full((E, M, H), 2.0)
#y = np.vstack([y1, y2])



x = tf.convert_to_tensor(x)
print("x shape ", x.shape)
#y = tf.convert_to_tensor(y)

#x_v = tf.gather(x, [0], axis=2)
#x_v = tf.gather(x, tf.where([True, False, True]), axis=0)

indices = np.array([
    [[[0,0,0],[0,0,1]]],
    [[[1,0,0],[1,0,1]]]
])
#,[[[1,1,1]]],[[[2,1,1]]]
x_v = tf.gather_nd(x, indices)


#y_v = tf.gather(y, [0,1], axis=0)



with tf.Session() as sess:
    print(sess.run(x))
    print("######## x_v: ")
    x_ = sess.run(x_v)
    print("x_ shape ", x_.shape)
    print(x_)
