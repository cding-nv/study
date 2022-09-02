import tensorflow as tf

x=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
y=tf.constant([[0,0,1.0],[0,0,1.0],[0,0,1.0]])

z = tf.matmul(x,y)

e = tf.einsum('ij,jk->ik',x,y)

with tf.Session() as sess:
    print(sess.run(z))
    print(sess.run(e))
