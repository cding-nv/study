import numpy as np
import tensorflow as tf
from my_modeling import attention_layer_orig, attention_layer_custom
# import sys
import time

# from __multi_head_attention_grad import construct_fw

ITER = 5
N = 16 # batch_size, 8000 in real case
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
lrelu_alpha = 0.2
attention_probs_dropout_prob=0.2

if is_fp16:
    TF_TYPE = tf.float16
    NP_TYPE = np.float16
    tolerance = 5e-3
else:
    TF_TYPE = tf.float32
    NP_TYPE = np.float32
    tolerance = 1e-6

seed = 1

class MultiHeadModel:
    """
    TF graph builder for the MultiHead Model.
    """
    def __init__(self,
                 batch_size,
                 batch_size_qv,
                 from_seq_len,
                 to_seq_len,
                 hidden_units,
                 size_per_head,
                 num_heads=8,
                 is_training=False):
        self.batch_size = batch_size
        self.batch_size_qv = batch_size_qv
        self.from_seq_len = from_seq_len
        self.to_seq_len = to_seq_len
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.size_per_head = size_per_head
        self.query = tf.placeholder(TF_TYPE,
                                    shape=[self.batch_size,
                                           self.from_seq_len,
                                           C_q])
        self.key = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size_qv,
                                         self.to_seq_len,
                                         C_k])
        self.value = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size_qv,
                                         self.to_seq_len,
                                         C_v])
        self.key_mask = tf.placeholder(tf.bool,
                                  shape=[self.batch_size_qv,
                                         self.to_seq_len])
        self.output0, self.output1= attention_layer_orig(
            queries=self.query,
            keys=self.key,
            values=self.value,
            key_mask=self.key_mask,
            num_attention_heads=self.num_heads,
            size_per_head=self.size_per_head,
            batch_size=self.batch_size,
            from_seq_length=self.from_seq_len,
            to_seq_length=self.to_seq_len,
            is_fp16=is_fp16,
            do_lrelu=do_leakyRelu,
            lrelu_alpha=lrelu_alpha,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

    def eval(self, np_query, np_key, np_value, np_key_mask):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run([self.output0, self.output1],
                               feed_dict={self.query:np_query,
                                          self.key:np_key,
                                          self.value:np_value,
                                          self.key_mask:np_key_mask
                                          })
            m = {}
            Model_variables = tf.GraphKeys.VARIABLES
            all_vars = tf.get_collection(Model_variables)
            for var in all_vars:
                data = sess.run(var)
                m[var.name] = data
                print(var.name)
                print(data.shape)
            # np.savez("./weight_bias.npz", **m)
            return outputs[0], outputs[1], m

class MyMultiHeadModel:
    def __init__(self,
                 batch_size,
                 batch_size_qv,
                 from_seq_len,
                 to_seq_len,
                 hidden_units,
                 weights,
                 num_heads=8):
        self.batch_size = batch_size
        self.batch_size_qv = batch_size_qv
        self.from_seq_len = from_seq_len
        self.to_seq_len = to_seq_len
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.query = tf.placeholder(TF_TYPE,
                                    shape=[self.batch_size,
                                           self.from_seq_len,
                                           C_q])
        self.key = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size_qv,
                                         self.to_seq_len,
                                         C_k])
        self.value = tf.placeholder(TF_TYPE,
                                  shape=[self.batch_size_qv,
                                         self.to_seq_len,
                                         C_v])
        self.key_mask = tf.placeholder(tf.bool,
                                  shape=[self.batch_size_qv,
                                         self.to_seq_len])

        self.output0, self.output1 = attention_layer_custom(
            queries=self.query,
            keys=self.key,
            values=self.value,
            key_mask=self.key_mask,
            num_attention_heads=self.num_heads,
            size_per_head=size_per_head,
            batch_size=self.batch_size,
            from_seq_length=self.from_seq_len,
            to_seq_length=self.to_seq_len,
            is_fp16=is_fp16,
            do_lrelu=do_leakyRelu,
            lrelu_alpha=lrelu_alpha,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )


    def eval(self, np_query, np_key, np_value, np_key_mask):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            my_outputs = sess.run([self.output0, self.output1],
                                  feed_dict={self.query:np_query,
                                             self.key:np_key,
                                             self.value:np_value,
                                             self.key_mask:np_key_mask
                                             })
        return my_outputs

#class TestMultiHeadModelFW:
#    def __init__(self,
#                 batch_size,
#                 batch_size_qv,
#                 from_seq_len,
#                 to_seq_len,
#                 hidden_units,
#                 weights,
#                 num_heads=8):
#        self.batch_size = batch_size
#        self.batch_size_qv = batch_size_qv
#        self.from_seq_len = from_seq_len
#        self.to_seq_len = to_seq_len
#        self.hidden_units = hidden_units
#        self.num_heads = num_heads
#        self.queries = tf.placeholder(TF_TYPE,
#                                    shape=[self.batch_size,
#                                           self.from_seq_len,
#                                           C_q])
#        self.keys = tf.placeholder(TF_TYPE,
#                                  shape=[self.batch_size_qv,
#                                         self.to_seq_len,
#                                         C_k])
#        self.values = tf.placeholder(TF_TYPE,
#                                  shape=[self.batch_size_qv,
#                                         self.to_seq_len,
#                                         C_v])
#        self.key_mask = tf.placeholder(tf.bool,
#                                  shape=[self.batch_size_qv,
#                                         self.to_seq_len])
#        attr_q_kernel = tf.convert_to_tensor(weights['query/kernel:0'])
#        attr_q_bias   = tf.convert_to_tensor(weights['query/bias:0'])
#        attr_k_kernel = tf.convert_to_tensor(weights['key/kernel:0'])
#        attr_k_bias   = tf.convert_to_tensor(weights['key/bias:0'])
#        attr_v_kernel = tf.convert_to_tensor(weights['value/kernel:0'])
#        attr_v_bias   = tf.convert_to_tensor(weights['value/bias:0'])
#
#        self.output = construct_fw(
#            self.queries,
#            self.keys,
#            self.values,
#            attr_q_kernel,
#            attr_q_bias,
#            attr_k_kernel,
#            attr_k_bias,
#            attr_v_kernel,
#            attr_v_bias,
#            self.key_mask,
#            self.num_heads,
#            self.hidden_units,
#            lrelu_alpha,
#            do_leakyRelu
#        )
#
#    def eval(self, np_query, np_key, np_value, np_key_mask):
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            my_outputs = sess.run(self.output,
#                                  feed_dict={self.queries:np_query,
#                                             self.keys:np_key,
#                                             self.values:np_value,
#                                             self.key_mask:np_key_mask
#                                             })
#        return my_outputs

def main():
    np_query = np.random.rand(N, T_q, C_q)
    np_query = np_query.astype(NP_TYPE)
    np_key = np.random.rand(M, T_k, C_k)
    np_key = np_key.astype(NP_TYPE)
    np_value = np.random.rand(M, T_k, C_v)
    np_value = np_value.astype(NP_TYPE)
    key_mask = np.random.choice([True], M*T_k)
    key_mask = key_mask.astype(np.bool)
    key_mask = key_mask.reshape(M, T_k)

    # TensorFlow implementation
    mha_model = MultiHeadModel(
                               batch_size=N,
                               batch_size_qv=M,
                               from_seq_len=T_q,
                               to_seq_len=T_k,
                               hidden_units=hidden_units,
                               size_per_head=size_per_head,
                               num_heads=num_heads
                               )
    ref_output0, ref_output1, weights = mha_model.eval(np_query, np_key, np_value, key_mask)

#
    tic0 = time.perf_counter()
    for _ in range(ITER):
         ref_output0, ref_output1, weights = mha_model.eval(np_query, np_key, np_value, key_mask)
    toc0 = time.perf_counter()

    print(ref_output0.shape)
    ref_output0 = ref_output0.reshape(-1)
    ref_output1 = ref_output1.reshape(-1)
    np.savetxt("ref_output0.txt", ref_output0[0:10240])
    np.savetxt("ref_output1.txt", ref_output1[0:10240])
    # np.savetxt("input.txt", ref_output1)

    # Customized implementation
    my_mha_model = MyMultiHeadModel(batch_size=N,
                                    batch_size_qv=M,
                                    from_seq_len=T_q,
                                    to_seq_len=T_k,
                                    hidden_units=hidden_units,
                                    weights=weights,
                                    num_heads=num_heads)
    my_outputs = my_mha_model.eval(np_query, np_key, np_value, key_mask)

    tic1 = time.perf_counter()
    for _ in range(ITER):
        my_outputs = my_mha_model.eval(np_query, np_key, np_value, key_mask)
    toc1 = time.perf_counter()

    my_output0 = my_outputs[0].reshape(-1)
    my_output1 = my_outputs[1].reshape(-1)
    np.savetxt("my_output0.txt", my_output0[0:10240])
    np.savetxt("my_output1.txt", my_output1[0:10240])

#    tf_api_model = TestMultiHeadModelFW(batch_size=N,
#                                        batch_size_qv=M,
#                                        from_seq_len=T_q,
#                                        to_seq_len=T_k,
#                                        hidden_units=hidden_units,
#                                        weights=weights,
#                                        num_heads=num_heads)
#    api_outputs = tf_api_model.eval(np_query, np_key, np_value, key_mask)
#
#    api_output0 = api_outputs[0].reshape(-1)
#    api_output1 = api_outputs[1].reshape(-1)

    # diff_output0 = ref_output0 - api_output0
    # diff_output1 = ref_output1 - api_output1
    # np.savetxt("diff_output_final.txt", diff_output1)
    # np.savetxt("diff_output_inter.txt", diff_output1)

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

    print("time for the ref TF: %.4f ms"%((toc0-tic0)/ITER*1000))
    print("time for the customied TF: %.4f ms"%((toc1-tic1)/ITER*1000))

    print("There are three different resuts:")
    print("1. Results from the original TensorFlow implementation")
    print("2. Result from the customized OP")
    print("3. Result from the Tensorflow APIs for backward computation")

    # compare the results
    print("Final output validity (1 vs 2):" + str(np.allclose(ref_output0, my_output0, atol = tolerance)))
    print("max absolute diff " + str(np.fabs(ref_output0 - my_output0).max()))
    print("mean absolute diff " + str(np.fabs(ref_output0 - my_output0).mean()))
    print("max relative diff " + str(np.fabs((ref_output0 - my_output0)/(ref_output0+1e-20)).max()))
    print("mean relative diff " + str(np.fabs((ref_output0 - my_output0)/(ref_output0+1e-20)).mean()))
    # print("min diff " + str(np.fabs(ref_output0 - my_output0).min()))

    print("Intermediate output validity (1 vs 2):" + str(np.allclose(ref_output1, my_output1, atol = tolerance)))
    print("max absolute diff " + str(np.fabs(ref_output1 - my_output1).max()))
    print("mean absolute diff " + str(np.fabs(ref_output1 - my_output1).mean()))
    print("max relative diff " + str(np.fabs((ref_output1 - my_output1)/(ref_output1+1e-20)).max()))
    print("mean relative diff " + str(np.fabs((ref_output1 - my_output1)/(ref_output1+1e-20)).mean()))


#    print("Intermediate validity (1 vs 2): " + str(np.allclose(ref_output1, my_output1, atol = tolerance)))
#    print("max diff " + str(np.fabs(ref_output1 - my_output1).max()))
#    # print("min diff " + str(np.fabs(ref_output1 - my_output1).min()))
#
#    print("Final output validity (1 vs 3): " + str(np.allclose(ref_output0, api_output0, atol = tolerance)))
#    print("max diff " + str(np.fabs(ref_output0 - api_output0).max()))
#    # print("min diff " + str(np.fabs(ref_output0 - api_output0).min()))
#
#    print("Intermediate validity (1 vs 3): " + str(np.allclose(ref_output1, api_output1, atol = tolerance)))
#    print("max diff " + str(np.fabs(ref_output1 - api_output1).max()))
#    # print("min diff " + str(np.fabs(ref_output1 - api_output1).min()))

    print("=============================================\n")

#    print("Compare more results:")
#    for i in range(1):
#        my_data = my_outputs[i].reshape(-1)
#        api_data = api_outputs[i].reshape(-1)
#        print("Results validity: " + str(np.allclose(my_data, api_data, atol = tolerance)))
#        print("max diff " + str(np.fabs(my_data - api_data).max()))
#


if __name__ == "__main__":
    main()
