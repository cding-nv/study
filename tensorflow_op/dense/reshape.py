import tensorflow as tf
input1 =[ [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]],

         [[[7, 7, 7], [8, 8, 8]],
         [[9, 9, 9], [10, 10, 10]],
         [[11, 11, 11], [12, 12, 12]]],

        [[[13, 13, 13], [14, 14, 14]],
         [[15, 15, 15], [16, 16, 16]],
         [[17, 17, 17], [18, 18, 18]]]
         ]

print(tf.shape(input1))
input2 = tf.reshape(input1,[-1,3])
print(tf.shape(input2))
index = tf.ones(shape=(2,3,2),dtype=tf.int32)
gather = tf.gather(input2,index)
with tf.Session() as sess:
    output1=tf.gather(input1, [0,2],axis=0)#其实默认axis=0
    print('input2')
    print(sess.run(input2))
    print('output1:')
    print(sess.run(output1))
    print('\n')
    print('index:\n')
    print(sess.run(index))
    print('gather:\n')
    print(sess.run(gather))
