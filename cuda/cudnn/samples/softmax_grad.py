import numpy as np
import tensorflow as tf
# import sys
# import time

B = 2
H = 16
S = 384
is_fp16 = False

if is_fp16:
    TF_TYPE = tf.float16
    NP_TYPE = np.float16
    tolerance = 5e-3
else:
    TF_TYPE = tf.float32
    NP_TYPE = np.float32
    tolerance = 1e-6

seed = 1

class test_backward:
    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 is_training=False):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads

        self.input = tf.placeholder(TF_TYPE, shape=[self.batch, self.heads, self.seq_len, self.seq_len])

        self.output = tf.nn.softmax(self.input, axis=-1)

    def backward_auto(self, np_key, batch, heads, seq_len):
        # grads = tf.placeholder(tf.float32,
        #                       shape=[
        #                              self.heads,
        #                              self.seq_len,
        #                              self.seq_len])
        g = tf.gradients(self.output, [self.input], grad_ys=np_key)  

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g_out = sess.run(g,
                             feed_dict={
                                 self.input:np_key
                                 })
        d_input= g_out
        return d_input

    def tf_custom_op(self, np_key_, batch, heads, seq_len):
        t_grads = tf.constant(np_key_, dtype=tf.float32)

        # load so
        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        custom_module = tf.load_op_library(custom_module_lib_file)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            s_out = sess.run(self.output,
                             feed_dict={
                                 self.input:np_key_
                                 })

        d_grad = custom_module.soft_max_grad(
            grad = t_grads,
            softmax_out = s_out,
            batch = self.batch,
            heads = self.heads,
            seq_len = self.seq_len
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            my_grad = sess.run([d_grad])

        print("### my_grad shape ", my_grad[0].shape)
        return my_grad[0]

def main():
    np_key = np.random.normal(size=(B, H, S, S))
    np_key = np_key.astype(np.float32)

    #np_key = np.ones((H, S, S), np.float32)

    model = test_backward(B,H,S)

    [ref_grad] = model.backward_auto(np_key, B, H, S)
    print("### ref softmax_grad shape = ", ref_grad.shape)
    np.savetxt("ref_softmax_grad.txt", ref_grad.reshape(-1))

    my_grad = model.tf_custom_op(np_key, B, H, S)
    #my_dk = tf.transpose(my_d_key, perm=(0, 2, 1))
    print("### My softmax_grad shape = ", my_grad.shape)
    np.savetxt("my_grad.txt", my_grad.reshape(-1))

if __name__ == "__main__":
    main()

