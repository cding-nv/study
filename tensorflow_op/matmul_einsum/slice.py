import tensorflow as tf

input = [
    [[1,1,1],[2,2,2]],
    [[3,3,3],[4,4,4]],
    [[5,5,5],[6,6,6]]
]

output = tf.slice(input, [1,0,0], [1,1,3])

with tf.Session() as sess:
    output = sess.run(output)
    print(output)

