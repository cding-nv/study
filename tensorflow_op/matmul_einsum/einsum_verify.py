import numpy as np
import tensorflow as tf

# input[0] (20, 1, 512, 1024)
# input[1] (20, 1024, 9984)


E = 2
G = 1
C = 2
M = 3

H = 2


x1 = np.full((E, G, C, M), 1.0)
x2 = np.full((E,G,C,M), 2.0)
x = np.vstack([x1, x2])

y1 = np.full((E, M, H), 3.0)
y2 = np.full((E, M, H), 2.0)
y = np.vstack([y1, y2])

x = tf.convert_to_tensor(x)
print("x shape ", x.shape)
y = tf.convert_to_tensor(y)



e = tf.einsum('EGCM,EMH->EGCH',x,y)


with tf.Session() as sess:
    x = sess.run(x)
    print("x shape ", x.shape)
    print(x)
    y = sess.run(y)
    print("y shape ", y.shape)
    print(y)
    print("######## e")
    print(sess.run(e))
    
