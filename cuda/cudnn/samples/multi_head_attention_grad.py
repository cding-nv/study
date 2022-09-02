import numpy as np
import tensorflow as tf
# import sys
import time,timeit

ITER = 1
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
                 seq_len_,
                 hidden_size_):
        # matrix 按列按顺序输入，所以需要转置一下
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

        # Only wo is different. cudnn sample saves it in order and then here read
        wo = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/wo_inorder.dat")
        #wo = np.transpose(wo)
        wo = wo.flatten()
        W = np.concatenate([wq, wk, wv, wo])
        W = W.astype(np.float32)
        #print("##W ", W)

        dout = np.loadtxt("/local/cudnn/cudnn_samples_v8/multiHeadAttention/dout.dat")
        dout = np.transpose(dout)
        dout = dout.flatten()
        dout = dout.astype(np.float32)

        self.batch = batch_
        self.heads = heads_
        self.seq_len = seq_len_
        self.hidden_size = hidden_size_
        print(" 111 ")
        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        custom_module = tf.load_op_library(custom_module_lib_file)
        print(" 222 ")
        self.my_dq, self.my_dk, self.my_dv, self.my_dw = custom_module.multi_head_attention_grad(
            input_q=Q,
            input_k=K,
            input_v=V,
            input_w=W,
            input_dout=dout,
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size)
        print(" 333 ")

    def eval(self):
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            my_outputs = sess.run([self.my_dq, self.my_dk, self.my_dv, self.my_dw])
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
    np.savetxt("my_devDQ.txt", my_output[0])

    print("my_output_0[1] shape ", my_output[1].shape)
    np.savetxt("my_devDK.txt", my_output[1])

    print("my_output_0[2] shape ", my_output[2].shape)
    np.savetxt("my_devDV.txt", my_output[2])

    print("my_output_0[3] shape ", my_output[3].shape)
    np.savetxt("my_devDW.txt", my_output[3])

    #print("my_output_0[4] shape ", my_output[4].shape)
    #np.savetxt("my_out.txt", my_output[4])

    #print("time for the ref TF: %.4f ms"%((toc0-tic0)/ITER*1000))
    print("time for the customied TF: %.4f ms"%((toc1-tic1)/ITER*1000))

if __name__ == "__main__":
    main()
