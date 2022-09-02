import numpy as np
import tensorflow as tf


a = np.arange(40).reshape(4, 10)
print(a)


#z = tf.split(a, 4, axis=0)
z0,z1,z2,z3 = np.split(a, 4, axis=0)
z = np.split(a, 4, axis=0)
z0_sum = np.sum(z[0])

print("z0_sum: ", z0_sum)

#with tf.Session() as sess:
#    print(sess.run(z))
#    print("z[0]: ", z[0])

#a1,a2,a3 = np.split(a, 3, axis=0)
#print(a1)
