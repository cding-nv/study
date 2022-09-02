import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)

tensor = [[1,1],[1,1],[1,1]]
indices = [[0,1],[2,0]]
updates = [5,10]

#indices = tf.constant([[0],[1],[2],[3],[4],[5],[6],[7]])
#updates = tf.constant(
#[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],
# [1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]
#]
#)
#shape = tf.constant([12,4])
#e_e = tf.scatter_nd(indices, updates, shape)

#print("updates shape ", updates.shape)


tensor1 = tf.zeros([6,3], dtype=tf.int32)
indices1 = tf.constant([[2],[4]])
updates1 = tf.constant([[1,2,3], [4,5,6]])

with tf.Session() as sess:
    e1 = tf.tensor_scatter_nd_update(tensor1, indices1, updates1)
    print(sess.run(e1))

    
