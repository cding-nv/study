import numpy as np

cc = np.array([[0, 0, 0],
 [0, 0, 3],
 [0, 0, 4],
 [0, 0, 7],
 [1, 0, 0],
 [1, 0, 1],
 [1, 0, 3],
 [1, 0, 4],
 [1, 0, 5],
 [1, 0, 6],
 [2, 0, 1],
 [2, 0, 5],
 [2, 0, 6],
 [2, 0, 7],
 [3, 0, 4],
 [3, 0, 5],
 [3, 0, 6],
 [3, 0, 7],
 [4, 0, 1],
 [4, 0, 2],
 [4, 0, 4],
 [4, 0, 5],
 [4, 0, 6],
 [4, 0, 7],
 [5, 0, 2],
 [5, 0, 3],
 [5, 0, 4],
 [5, 0, 5],
 [6, 0, 0],
 [6, 0, 2],
 [6, 0, 4],
 [6, 0, 6],
 [6, 0, 7],
 [7, 0, 0],
 [7, 0, 1],
 [7, 0, 2],
 [7, 0, 5],
 [7, 0, 6],
 [7, 0, 7],
 [8, 0, 2],
 [8, 0, 3],
 [8, 0, 5],
 [8, 0, 6],
 [9, 0, 1],
 [9, 0, 3],
 [9, 0, 5],
 [9, 0, 6],
 [9, 0, 7]])

#for i in range(cc.shape[0]):
#    position_v_pad[i] = cc[i][0] * C + cc[i][2]

print(cc[:, 0:1]*10 + cc[:, 2:3])

#with tf.Session() as sess:
