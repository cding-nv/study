import numpy as np
import tensorflow as tf
import __multi_head_attention_grad
# import __multi_head_attention_grad_auto_grad
# import __multi_head_attention_grad_tf_apis
from my_modeling import attention_layer_orig, attention_layer_custom
# import sys
# import time

# from __multi_head_attention_grad import construct_fw
# NOTES:
# Backward of softmax introduces large error
# Backward of matmul (after ln) introduces large error (1e-6 -> 1e-5)

ITER = 5
N = 16 # batch_size
M = N  # in Alimama's new model, M == N
T_q = 384  # sequence length
T_k = 384
C_q = 1024
C_k = 1024
C_v = 1024
C = 1024 # hidden dimension = head_num * size_per_head
num_heads=16
size_per_head = C//num_heads
hidden_units=C
is_fp16 =False
do_leakyRelu = True
lrelu_alpha = 0.20
dropout_rate=0.0

if is_fp16:
    TF_TYPE = tf.float16
    NP_TYPE = np.float16
    tolerance = 5e-3
else:
    TF_TYPE = tf.float32
    NP_TYPE = np.float32
    tolerance = 1e-3

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=lrelu_alpha)

def leaky_relu_back(d_out, x_input):
    d_new = tf.where((x_input>0), d_out, lrelu_alpha*d_out)
    return d_new

class TestMultiHeadModel:
    def __init__(self,
                 batch_size,
                 from_seq_len,
                 to_seq_len,
                 hidden_units,
                 size_per_head,
                 num_heads=8):
        self.batch_size = batch_size
        self.from_seq_len = from_seq_len
        self.to_seq_len = to_seq_len
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.size_per_head = size_per_head

        self.queries = tf.placeholder(TF_TYPE,
                                    shape=[self.batch_size,
                                           self.from_seq_len,
                                           C_q])
        self.keys = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size,
                                         self.to_seq_len,
                                         C_k])
        self.values = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size,
                                         self.to_seq_len,
                                         C_v])
        self.key_mask = tf.placeholder(tf.bool,
                                  shape=[self.batch_size,
                                         self.to_seq_len])
        self.output,_ = attention_layer_orig(
            queries=self.queries,
            keys=self.keys,
            values=self.values,
            key_mask=self.key_mask,
            num_attention_heads=self.num_heads,
            size_per_head=self.size_per_head,
            batch_size=self.batch_size,
            from_seq_length=self.from_seq_len,
            to_seq_length=self.to_seq_len,
            is_fp16=is_fp16,
            do_lrelu=do_leakyRelu,
            lrelu_alpha=lrelu_alpha,
            attention_probs_dropout_prob=dropout_rate
        )

    def eval(self, np_query, np_key, np_value, np_key_mask):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            my_outputs= sess.run(self.output,
                                  feed_dict={self.queries:np_query,
                                             self.keys:np_key,
                                             self.values:np_value,
                                             self.key_mask:np_key_mask
                                             })
        return my_outputs

    def backprop(self, np_grads, np_query, np_key, np_value, np_key_mask):

        self.grads = tf.placeholder(tf.float32,
                               shape=[self.batch_size,
                                      self.from_seq_len,
                                      self.hidden_units])

        g = tf.gradients(self.output, [self.queries,
                                       self.keys,
                                       self.values
                                       ],
                         grad_ys=self.grads)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            g_out = sess.run(g,
                             feed_dict={
                                 self.queries:np_query,
                                 self.keys:np_key,
                                 self.values:np_value,
                                 self.key_mask:np_key_mask,
                                 self.grads:np_grads
                             })
        return g_out


class MyMultiHeadModel:
    def __init__(self,
                 batch_size,
                 from_seq_len,
                 to_seq_len,
                 hidden_units,
                 size_per_head,
                 num_heads=8):
        self.batch_size = batch_size
        self.from_seq_len = from_seq_len
        self.to_seq_len = to_seq_len
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.size_per_head = size_per_head

        self.queries = tf.placeholder(TF_TYPE,
                                    shape=[self.batch_size,
                                           self.from_seq_len,
                                           C_q])
        self.keys = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size,
                                         self.to_seq_len,
                                         C_k])
        self.values = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size,
                                         self.to_seq_len,
                                         C_v])
        self.key_mask = tf.placeholder(tf.bool,
                                  shape=[self.batch_size,
                                         self.to_seq_len])
        self.output,_ = attention_layer_custom(
            queries=self.queries,
            keys=self.keys,
            values=self.values,
            key_mask=self.key_mask,
            num_attention_heads=self.num_heads,
            size_per_head=self.size_per_head,
            batch_size=self.batch_size,
            from_seq_length=self.from_seq_len,
            to_seq_length=self.to_seq_len,
            is_fp16=is_fp16,
            do_lrelu=do_leakyRelu,
            lrelu_alpha=lrelu_alpha,
            attention_probs_dropout_prob=dropout_rate
        )

    def eval(self, np_query, np_key, np_value, np_key_mask):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            my_outputs= sess.run(self.output,
                                  feed_dict={self.queries:np_query,
                                             self.keys:np_key,
                                             self.values:np_value,
                                             self.key_mask:np_key_mask
                                             })
        return my_outputs

    def backprop(self, np_grads, np_query, np_key, np_value, np_key_mask):

        self.grads = tf.placeholder(tf.float32,
                               shape=[self.batch_size,
                                      self.from_seq_len,
                                      self.hidden_units])

        g = tf.gradients(self.output, [self.queries,
                                       self.keys,
                                       self.values
                                       ],
                         grad_ys=self.grads)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            g_out = sess.run(g,
                             feed_dict={
                                 self.queries:np_query,
                                 self.keys:np_key,
                                 self.values:np_value,
                                 self.key_mask:np_key_mask,
                                 self.grads:np_grads
                             })
        return g_out


def main():
    # np.random.seed(1234)
    np_query = np.random.rand(N, T_q, C_q)
    np_query = np_query.astype(NP_TYPE)
    np_key = np.random.rand(M, T_k, C_k)
    np_key = np_key.astype(NP_TYPE)
    np_value = np.random.rand(M, T_k, C_v)
    np_value = np_value.astype(NP_TYPE)
    np_grads = np.random.rand(N, T_q, num_heads*size_per_head)
    np_grads = np_grads.astype(NP_TYPE)
    key_mask = np.random.choice([True], M*T_k)
    key_mask = key_mask.astype(np.bool)
    key_mask = key_mask.reshape(M, T_k)

    # TensorFlow implementation
    tf_model = TestMultiHeadModel(batch_size=N,
                                  from_seq_len=T_q,
                                  to_seq_len=T_k,
                                  hidden_units=hidden_units,
                                  size_per_head=size_per_head,
                                  num_heads=num_heads)

    # outputs = tf_model.eval( np_query, np_key, np_value, key_mask)

    for i in range(ITER):
        auto_tmp = tf_model.backprop(np_grads, np_query, np_key, np_value, key_mask)

    backward_model = MyMultiHeadModel(
        batch_size=N,
        from_seq_len=T_q,
        to_seq_len=T_k,
        hidden_units=hidden_units,
        size_per_head=size_per_head,
        num_heads=num_heads
    )

    for i in range(ITER):
        my_tmp = backward_model.backprop( np_grads, np_query, np_key, np_value, key_mask)


    names = ["dq", "dk", "dv"]

    print("\n")
    print("=============================================")
    if is_fp16:
        print("RUNNING test in FP16 mode.")
    else:
        print("RUNNING test in FP32 mode.")
    print('batch_size (N): %d' % N)
    print('seq len T_q: %d, T_k: %d' % (T_q, T_k))
    print('hidden size (C_q): %d' % C_q)
    print('hidden size (C_k): %d' % C_k)
    print('hidden size (C): %d' % C)
    print("\n")

    for i in range(3):
        my_data = my_tmp[i].reshape(-1)
        tf_data = auto_tmp[i].reshape(-1)
        np.savetxt('my_data_{}.txt'.format(i), my_data[0:1024])
        np.savetxt('auto_data_{}.txt'.format(i), tf_data[0:1024])

    print("Custom OP vs TF API:")
    for i in range(3):
        my_grad = my_tmp[i].reshape(-1)
        tf_grad = auto_tmp[i].reshape(-1)
        print("interation id: " + str(i) + ", grad: " + names[i])
        # print("Comparison:" + str(np.allclose(my_grad, tf_grad, atol = 5e-6)))
        # print("max diff " + str(np.fabs(my_grad - tf_grad).max()))
        print("mean rel err " + str(np.fabs((my_grad - tf_grad)/(tf_grad + 1e-20)).mean()))

    print("=============================================\n")

if __name__ == "__main__":
    main()


