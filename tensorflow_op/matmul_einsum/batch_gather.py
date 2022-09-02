import tensorflow as tf

#元の点群(2batch x 8点 x 3次元)
param = tf.constant([[[0.,0.,9.],[0.,1.,9.], [0.,2.,9.], [0.,3.,9.], [0.,4.,9.], [0.,5.,9.], [0.,6.,9.], [0.,7.,9.]],
                     [[1.,0.,9.],[1.,1.,9.], [1.,2.,9.], [1.,3.,9.], [1.,4.,9.], [1.,5.,9.], [1.,6.,9.], [1.,7.,9.]]]) # B x N x C(2 x 8 x 3)
print("param shape:     ",param.shape)

#gather元を選ぶテンソル(2batch x 4点)
indices = tf.constant([[1,0,0,4],
                       [1,3,4,6]])    # B x N
print("indices shape:   ",indices.shape)

#gatherする
result = tf.batch_gather(param, indices)
print("result shape:    ",result.shape)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("param\n",sess.run(param))     #入力の確認
print("indices\n",sess.run(indices)) #indiciesの確認
print("result\n",sess.run(result))   #gatherの結果
