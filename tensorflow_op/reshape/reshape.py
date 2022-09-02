import tensorflow as tf
import numpy as np
#from tensorflow.keras import layers
tf.enable_eager_execution()

#t1 = np.full((2, 3, 3), 1.0)
#t2 = np.full((2, 3, 3), 1.0)

t1 = tf.constant([[1,2,3,4],[1, 0, 0, 0],[2,2,2,2]])
t2 = tf.constant([[4,5,6],[1, 1, 1]])

width = t1.shape[-1]

print(width)

t1 = tf.reshape(t1, [-1])
t2 = tf.reshape(t2, [-1])

print(t1)
print(t2)

