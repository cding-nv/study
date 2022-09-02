import numpy as np
import tensorflow as tf 
  
with tf.Session() as sess:
     output = tf.where([True, False, True, False, True, True])
     print(sess.run(output))
     #print("list: ", output)
     #print("list :", list(output))
     output = tf.reshape(output, [-1])
     res = sess.run(output)
     print("res: ")
     print(res)
     print("list: ", list(res))
