import tensorflow as tf
tf.enable_eager_execution()
ones = tf.ones(shape=[1,2,8])
temp1 = tf.nn.softmax(ones,axis=0) # 列
print(temp1)
temp2 = tf.nn.softmax(ones,axis=1) # 行
print(temp2)
