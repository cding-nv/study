import numpy as np
import tensorflow as tf
# import sys
import time,timeit

ITER = 1
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
                 seq_len_,
                 hidden_size_):
        # matrix is saved in column, so needs transpose.
        Q = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/q.dat")
        Q = np.transpose(Q)
        Q = Q.flatten()
        Q = Q.astype(np.float32)
        #print("q: ", Q)
        K = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/k.dat")
        K = np.transpose(K)
        K = K.flatten()
        K = K.astype(np.float32)
        V = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/v.dat")
        V = np.transpose(V)
        V = V.flatten()
        V = V.astype(np.float32)
        wq = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/wq.dat")
        wq = np.transpose(wq)
        wq = wq.flatten()
        wq = wq.astype(np.float32)
        wk = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/wk.dat")
        wk = np.transpose(wk)
        wk = wk.flatten()
        wk = wk.astype(np.float32)
        wv = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/wv.dat")
        wv = np.transpose(wv)
        wv = wv.flatten()
        wv = wv.astype(np.float32)
        wo = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/wo_inorder.dat")
        #wo = np.transpose(wo)
        wo = wo.flatten()
        W = np.concatenate([wq, wk, wv, wo])
        W = W.astype(np.float32)
        #print("##W ", W)
        self.batch = batch_
        self.heads = heads_
        self.seq_len = seq_len_
        self.hidden_size = hidden_size_

        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        print("####test1")
        custom_module = tf.load_op_library(custom_module_lib_file)
        print("####test2")
        self.my_output = custom_module.multi_head_attention(
            input_q=Q,
            input_k=K,
            input_v=V,
            input_w=W,
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            hiddensize=self.hidden_size)

    def eval(self):
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            my_outputs = sess.run([self.my_output])
        return my_outputs

def main():
    
    

    # TensorFlow implementation
    #s_model = SoftmaxModel(batch=B, heads=H, seq_len=S)

    #tic0 = timeit.default_timer()
    #for _ in range(ITER):
    #    ref_output = s_model.eval(np_input)
    #toc0 = timeit.default_timer()
   
    #ref_output_0 = ref_output[0].reshape(-1)
    #np.savetxt("ref_output.txt", ref_output_0)
    #print('### ref_output[0].shape: ', ref_output[0].shape)

    # Customized implementation
    my_model = MySoftmaxModel(batch_ = 1, heads_ = 3, seq_len_= 4, hidden_size_ = 8)
    
    tic1 = timeit.default_timer()
    for _ in range(ITER):
        my_output = my_model.eval()
    toc1 = timeit.default_timer()

    #my_output_0 = my_output[0].reshape(-1)
    print("my_output_0[0] shape ", my_output[0].shape)
    np.savetxt("my_output.txt", my_output[0])

    #print("time for the ref TF: %.4f ms"%((toc0-tic0)/ITER*1000))
    print("time for the customied TF: %.4f ms"%((toc1-tic1)/ITER*1000))

if __name__ == "__main__":
    main()
