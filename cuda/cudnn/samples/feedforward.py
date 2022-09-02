import numpy as np
import tensorflow as tf
from my_modeling import attention_layer
# import sys
import time,timeit

ITER = 2
N = 1 # batch_n
M = 4096 
T_q = 1  # sequence length
T_k = 150
C = 8  # hidden dimension = head_num * size_per_head
num_heads=2
D_Q_0 = 4
D_Q_1 = 32
D_K_0 = 32
num_units = C

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

class MultiHeadModel:
    """
    TF graph builder for the MultiHead Model.
    """
    def __init__(self,
                 batch_n,
                 batch_m,
                 from_seq_len,
                 to_seq_len,
                 num_units,
                 num_heads=8,
                 is_training=False):
        self.batch_n = batch_n
        self.batch_m = batch_m
        self.from_seq_len = from_seq_len
        self.to_seq_len = to_seq_len
        self.num_heads = num_heads
        self.num_units = num_units
        # 8192,1,96
        print('###self.batch_n/from_seq_len/num_units=', self.batch_n, '/', self.from_seq_len, '/', self.num_units)
        self.query = tf.placeholder(TF_TYPE,
                                    shape=[self.batch_n,
                                           self.from_seq_len,
                                           self.num_units])
        #self.key = tf.placeholder(TF_TYPE,
        #                          shape=[self.batch_n,
        #                                 self.to_seq_len,
        #                                 self.num_units])
        #self.indices = tf.placeholder(tf.int32,
        #                          shape=[self.batch_n])
        self.output = attention_layer(
            query=self.query,
            #key=self.key,
            #indices=self.indices,
            D_Q_0=D_Q_0,
            D_Q_1=D_Q_1,
            D_K_0=D_K_0,
            head_num=self.num_heads,
            is_fp16=is_fp16
        )

    def eval(self, np_query, np_key, np_indices):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run([self.output],
                               feed_dict={self.query:np_query})
            m = {}
            Model_variables = tf.GraphKeys.VARIABLES
            all_vars = tf.get_collection(Model_variables)
            for var in all_vars:
                data = sess.run(var)
                m[var.name] = data
                print('Peak var.name: ', var.name)
                print('Peak data shape: ', data.shape)
            # np.savez("./weight_bias.npz", **m)
            return outputs, m

class MyMultiHeadModel:
    def __init__(self,
                 batch_n,
                 batch_m,
                 from_seq_len,
                 to_seq_len,
                 num_units,
                 weights,
                 num_heads=8):
        self.batch_n = batch_n
        self.batch_m = batch_m
        self.from_seq_len = from_seq_len
        self.to_seq_len = to_seq_len
        self.num_units = num_units
        self.num_heads = num_heads
        print("### MyMultiHeadModel queries shape ", self.batch_n, self.from_seq_len, self.num_units)
        self.queries = tf.placeholder(TF_TYPE,
                                    shape=[self.batch_n,
                                           self.from_seq_len,
                                           self.num_units])
        # self.keys = tf.placeholder(TF_TYPE,
        #                           shape=[self.batch_n,
        #                                  self.to_seq_len,
        #                                  self.num_units])
        #self.indices = tf.placeholder(tf.int32,
        #                          shape=[self.batch_n])
        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        custom_module = tf.load_op_library(custom_module_lib_file)
        attr_q_kernel_0 = tf.convert_to_tensor(weights['attr/query_0/kernel:0'])
        #attr_q_bias_0   = tf.convert_to_tensor(weights['attr/query_0/bias:0'])
        #attr_k_kernel = tf.convert_to_tensor(weights['attr/key/kernel:0'])
        #attr_k_bias   = tf.convert_to_tensor(weights['attr/key/bias:0'])
        #attr_q_kernel_1 = tf.convert_to_tensor(weights['attr/query_1/kernel:0'])
        #attr_q_bias_1   = tf.convert_to_tensor(weights['attr/query_1/bias:0'])
        self.my_output = custom_module.feed_forward(
            from_tensor=self.queries,
            attr_q_kernel_0=attr_q_kernel_0,
            attr_q_bias_0=attr_q_kernel_0,
            head_num=self.num_heads,
            D_Q_0=D_Q_0,
        )

    def eval(self, np_query, np_key, np_indices):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            my_outputs = sess.run([self.my_output], feed_dict={self.queries:np_query})
        return my_outputs

def main():
    #np_query = np.random.rand(N, T_q, C)
    
    np.set_printoptions(precision=1)
    np_query = np.full((1,1,8), 1.0)
    np_query = np_query.astype(NP_TYPE)
    np_key = np.random.rand(N, T_k, C)
    np_key = np_key.astype(NP_TYPE)
    np_indices = np.random.randint(0, M, size=N, dtype=np.int32)

    # TensorFlow implementation
    mha_model = MultiHeadModel(
                               batch_n=N,
                               batch_m=M,
                               from_seq_len=T_q,
                               to_seq_len=T_k,
                               num_units=num_units,
                               num_heads=num_heads
                               )
    ref_output, weights = mha_model.eval(np_query, np_key, np_indices)

    tic0 = timeit.default_timer()
    for _ in range(ITER):
         ref_output, weights = mha_model.eval(np_query, np_key, np_indices)
    toc0 = timeit.default_timer()

    print('Peak ref_output[0].shape: ', ref_output[0].shape)
    ref_output_0 = ref_output[0].reshape(-1)
    #ref_output_1 = ref_output[1].reshape(-1)
    #ref_output_2 = ref_output[2].reshape(-1)
    #ref_output_3 = ref_output[3].reshape(-1)
    #ref_output_4 = ref_output[4].reshape(-1)
    np.savetxt("ref_output.txt", ref_output_0)

    weights_1 = weights["attr/query_0/kernel:0"].reshape(-1)
    np.savetxt("weight_kernel.txt", weights_1)

    query_ = np_query.reshape(-1)
    np.savetxt("query.txt", query_)

    # Customized implementation
    my_mha_model = MyMultiHeadModel(batch_n=N,
                                    batch_m=M,
                                    from_seq_len=T_q,
                                    to_seq_len=T_k,
                                    num_units=num_units,
                                    weights=weights,
                                    num_heads=num_heads)
    print('Peak after MyMultiHeadModel')
    my_output = my_mha_model.eval(np_query, np_key, np_indices)
    print('Peak end of my_mha_model.eval')

    #tic1 = timeit.default_timer()
    #for _ in range(ITER):
    #    my_output = my_mha_model.eval(np_query, np_key, np_indices)
    #toc1 = timeit.default_timer()

    my_output_0 = my_output[0].reshape(-1)
    print("my_output_0[0] shape ", my_output[0].shape)
    #my_output_1 = my_output[1].reshape(-1) # alphas
    #my_output_2 = my_output[2].reshape(-1) # query_1
    #my_output_3 = my_output[3].reshape(-1) # query_2
    #my_output_4 = my_output[4].reshape(-1) # key_1
    np.savetxt("my_output.txt", my_output_0)

if __name__ == "__main__":
    main()
