import numpy as np
import tensorflow as tf
# import sys
import time,timeit

ITER = 10
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

class SoftmaxModel:
    """
    TF graph builder for the MultiHead Model.
    """
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

    def eval(self, np_input):
        with tf.Session() as sess:
            outputs = sess.run([self.output],
                               feed_dict={self.input:np_input})
            return outputs

class MySoftmaxModel:
    def __init__(self,
                 batch_,
                 heads_,
                 seq_len_):
        self.batch = batch_
        self.heads = heads_
        self.seq_len = seq_len_
        self.input = tf.placeholder(TF_TYPE,
                                    shape=[self.batch, self.heads, self.seq_len, self.seq_len])

        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        custom_module = tf.load_op_library(custom_module_lib_file)

        self.my_output = custom_module.soft_max(
            from_tensor=self.input,
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len)

    def eval(self, np_input):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            my_outputs = sess.run([self.my_output], feed_dict={self.input:np_input})
        return my_outputs

def main():
    np_input1 = np.full((1, H, S, S), 1.0)
    np_input2 = np.full((1, H, S, S), 2.0)
    np_input = np.vstack([np_input1, np_input2])
    np_input = np_input.astype(NP_TYPE)
    print("########################", np_input.shape)

    # TensorFlow implementation
    s_model = SoftmaxModel(batch=B, heads=H, seq_len=S)

    tic0 = timeit.default_timer()
    for _ in range(ITER):
        ref_output = s_model.eval(np_input)
    toc0 = timeit.default_timer()
   
    ref_output_0 = ref_output[0].reshape(-1)
    np.savetxt("ref_output.txt", ref_output_0)
    print('### ref_output[0].shape: ', ref_output[0].shape)

    # Customized implementation
    my_model = MySoftmaxModel(batch_ = B, heads_ = H, seq_len_= S)
    
    tic1 = timeit.default_timer()
    for _ in range(ITER):
        my_output = my_model.eval(np_input)
    toc1 = timeit.default_timer()

    my_output_0 = my_output[0].reshape(-1)
    print("my_output_0[0] shape ", my_output[0].shape)
    np.savetxt("my_output.txt", my_output_0)

    print("time for the ref TF: %.4f ms"%((toc0-tic0)/ITER*1000))
    print("time for the customied TF: %.4f ms"%((toc1-tic1)/ITER*1000))

if __name__ == "__main__":
    main()
