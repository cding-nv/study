import tensorflow as tf 
  
with tf.Session() as sess:
     #output = tf.where([True, False, True, False, True, True]) 
     output = tf.where([1, 0, 1, 0, 1, 1]) 
     output = tf.reshape(output, [-1])
     res = sess.run(output)
     print("res: ")
     print(res)
