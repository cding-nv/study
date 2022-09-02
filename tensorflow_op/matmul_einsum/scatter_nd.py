import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)

indices = tf.constant([[0,0,1], [0,0,0],[0,0,2],[0,0,3],
                       [1,0,1], [1,0,0],[1,0,2],[1,0,3]])
#indices = tf.constant([[[0],[1],[2],[3]],[[0],[1],[2],[3]]])
updates = tf.constant(
[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],
 [1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]
]
)
shape = tf.constant([2,4,4,4])
e_e = tf.scatter_nd(indices, updates, shape)

print("updates shape ", updates.shape)


shape2 = tf.constant([10, 1, 1, 4])
e_e2 =  tf.scatter_nd([])

with tf.Session() as sess:
    print("######## e_e")
    print(sess.run(e_e))
    
