import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)


E = 2
G = 1
C = 4
M = 3

H = 2


data = np.array([
    [[1,1,1],[2,2,2]],
    [[3,3,3],[4,4,4]],
    [[5,5,5],[6,6,6]]
])
print("data shape ", data.shape)

#indices = np.array([[0,1],[1,0]])
indices = np.array([
    [[0,1]]
])


x_v = tf.gather_nd(data, indices)

with tf.Session() as sess:
    #print(sess.run(x))
    print("######## x_v: ")
    x_ = sess.run(x_v)
    print("x_ shape ", x_.shape)
    print(x_)
