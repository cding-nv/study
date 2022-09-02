import tensorflow as tf

sess=tf.Session()
a=tf.get_variable("a",[3,3,32,64],initializer=tf.random_normal_initializer())
b=tf.get_variable("b",[64],initializer=tf.random_normal_initializer())
#collections=None等价于 collection=[tf.GraphKeys.GLOBAL_VARIABLES]
print("I am a:", a)
print("I am b:", b)
print("I am gv:", tf.GraphKeys.GLOBAL_VARIABLES)
gv= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
          #tf.get_collection(collection_name)返回某个collection的列表


print("I am gv:", gv)
for var in gv:
    print("Iam var:",var)
    print(var is a)
    print(var.get_shape())
    print("----------------")
