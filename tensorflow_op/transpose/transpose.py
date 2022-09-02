import tensorflow as tf
import numpy as np
 
A = np.arange(12).reshape([2,3,2])
X = tf.transpose(A,[0,2,1]) # 交换维度 1 和维度 2
Y = tf.transpose(A,[1,0,2]) # 交换维度 0 和维度 1 
with tf.Session() as sess:
    print('original:\n', A)
    print('A.shape：', A.shape)
    print('='*30)
    
    print('transpose [0,2,1]:\n', sess.run(X))   
    print('X.shape：', X.shape)
    print('='*30)
    
    print('transpose [1,0,2]:\n', sess.run(Y))
    print('Y.shape：', Y.shape)
