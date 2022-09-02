#https://blog.csdn.net/kkxi123456/article/details/103739404


# Importing the library 
import tensorflow as tf 
  
input =[ [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]],

         [[[7, 7, 7], [8, 8, 8]],
         [[9, 9, 9], [10, 10, 10]],
         [[11, 11, 11], [12, 12, 12]]],

        [[[13, 13, 13], [14, 14, 14]],
         [[15, 15, 15], [16, 16, 16]],
         [[17, 17, 17], [18, 18, 18]]]
         ]

input1 = [
    [[[1,1]]],
    [[[2,2]]],
    [[[3,3]]]
]

with tf.Session() as sess:
  output1=tf.scatter_nd([[0], [1], [2]], input1, [5, 1, 1, 2])#其实默认axis=0
  print("output1:", output1)
  res = sess.run(output1)
  print("res: ", res)


