import numpy as np
import tensorflow as tf
# import sys
import time,timeit

ITER = 10
B = 4
#H = 2
S = 6
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

class LayernormModel:
    """
    TF graph builder for the MultiHead Model.
    """
    def __init__(self,
                 batch,
                 #heads,
                 seq_len,                
                 is_training=False):
        self.batch = batch
        self.seq_len = seq_len
        #self.heads = heads

        #self.input = tf.placeholder(TF_TYPE, shape=[self.batch, self.heads, self.seq_len])
        self.input = tf.placeholder(TF_TYPE, shape=[self.batch, self.seq_len])

        #self.output = tf.nn.softmax(self.input, axis=1)
        self.output = tf.contrib.layers.layer_norm(inputs=self.input, begin_norm_axis=-1, begin_params_axis=-1)

    def eval(self, np_input):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run([self.output],
                               feed_dict={self.input:np_input})
            return outputs

class MyLayernormModel:
    def __init__(self,
                 batch_,
                 #heads_,
                 seq_len_):
        self.batch = batch_
        #self.heads = heads_
        self.seq_len = seq_len_
        self.input = tf.placeholder(TF_TYPE,
                                    shape=[self.batch, self.seq_len])

        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        custom_module = tf.load_op_library(custom_module_lib_file)

        self.my_output = custom_module.layer_norm(
            from_tensor=self.input,
            batch=self.batch,
            #heads=1,
            seq_len=self.seq_len)

    def eval(self, np_input):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            my_outputs = sess.run([self.my_output], feed_dict={self.input:np_input})
        return my_outputs

def main():
    #np_input = np.full((B, H, S), 1.0)
    #np_input = np_input.astype(NP_TYPE)

    x_input = [[[18.369314, 2.6570225, 20.402943],
                [10.403599, 2.7813416, 20.794857]],
               [[19.0327, 2.6398268, 6.3894367],
                [3.921237, 10.761424, 2.7887821]],
               [[11.466338, 20.210938, 8.242946],
                [22.77081, 11.555874, 11.183836]],
               [[8.976935, 10.204252, 11.20231],
                [-7.356888, 6.2725096, 1.1952505]]]

    y_input = [[18.369314, 2.6570225, 20.402943, 10.403599, 2.7813416, 20.794857],
               [19.0327, 2.6398268, 6.3894367, 3.921237, 10.761424, 2.7887821],
               [11.466338, 20.210938, 8.242946, 22.77081, 11.555874, 11.183836],
               [8.976935, 10.204252, 11.20231, -7.356888, 6.2725096, 1.1952505]]

    # TensorFlow implementation
    l_model = LayernormModel(batch=B, seq_len=S)

    tic0 = timeit.default_timer()
    for _ in range(ITER):
        ref_output = l_model.eval(y_input)
    toc0 = timeit.default_timer()
   
    ref_output_0 = ref_output[0].reshape(-1)
    np.savetxt("ref_output.txt", ref_output_0)
    print('### ref_output[0].shape: ', ref_output[0].shape)

    # Customized implementation
    my_model = MyLayernormModel(batch_ = B, seq_len_= S)
    
    tic1 = timeit.default_timer()
    for _ in range(ITER):
        my_output = my_model.eval(y_input)
    toc1 = timeit.default_timer()

    my_output_0 = my_output[0].reshape(-1)
    print("my_output_0[0] shape ", my_output[0].shape)
    np.savetxt("my_output.txt", my_output_0)

    print("time for the ref TF: %.4f ms"%((toc0-tic0)/ITER*1000))
    print("time for the customied TF: %.4f ms"%((toc1-tic1)/ITER*1000))

if __name__ == "__main__":
    main()
