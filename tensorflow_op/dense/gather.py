import tensorflow as tf 
  
# Initializing the input 
data = tf.constant([1, 2, 3, 4, 5, 6]) 
indices = tf.constant([0, 1, 2, 1]) 
  
# Printing the input 
print('data:',data) 
print('indices:',indices) 
  
# Calculating result 
res = tf.gather(data, indices) 
  
# Printing the result 
print('res:',res)
