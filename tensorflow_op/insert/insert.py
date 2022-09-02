import tensorflow as tf
import numpy as np

a = np.array([
    [[[1,2,3]]],
    [[[4,5,6]]],
    [[[7,8,9]]]
])

a = tf.convert_to_tensor(a)

#sess = tf.session()
with tf.Session() as sess:
    a = a.eval()

b = np.array([[[[0]*3]]])

c = np.insert(a, 0, b, axis=0)

print("a shape ", a.shape)
print("b shape ", b.shape)
print("c shape ", c.shape)
print(c)

