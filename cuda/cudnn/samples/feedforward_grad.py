import numpy as np
import tensorflow as tf
# import sys
# import time

h = 1
M = 2  # out of memory for 1
N = 2
S = 8
DK0 = 8
DQ0 = 64
DQ1 = DK0
L = 3

#    model = test_backward(M, N, S, h, DK0, DK0, DQ1, L) 

class test_backward:
    def __init__(self,
                 batch_size,
                 bs_big,
                 seq_len,
                 head_num,
                 dk0,
                 dq0,
                 dq1,
                 L
                 ):
        self.batch_size = batch_size
        self.bs_big = bs_big
        self.seq_len = seq_len
        self.dk0 = dk0
        self.dq0 = dq0
        self.dq1 = dq1
        self.head_num = head_num
        self.hidden_size = L

        # [M, S, L]
        self.key = tf.placeholder(tf.float32,
                                     shape=[
                                            self.batch_size,
                                            self.seq_len,
                                            self.hidden_size
                                            ])
        print("###Peak: input key shape: ", self.batch_size, "/", self.seq_len,  "/", self.hidden_size)

        # [L, h*dk0]
        self.w_k0 = tf.placeholder(tf.float32,
                                   shape=[
                                       self.hidden_size,
                                       self.head_num * self.dk0
                                   ])
        print("###Peak: input w shape: ", self.hidden_size, "/", self.head_num * self.dk0)

        # [h*dk0]
        self.b0 = tf.placeholder(tf.float32,
                                 shape=[self.head_num * self.dk0])
        print("###Peak: input b shape: ", self.head_num * self.dk0)                      

        # [N]
        self.indices = tf.placeholder(tf.int32,
                                      shape=[self.bs_big])

        # [M, 1, h*dk0]
        t_key = tf.matmul(self.key, self.w_k0)

        # [M, 1, h*dk0]
        t_key = t_key + self.b0

        # [N, 1, h*dk0]
        #t_key = tf.gather(t_key, self.indices)

        # [h *N, 1, dk0]
        #out = tf.concat(tf.split(t_key, self.head_num, axis=2), axis=0)
        out = t_key
        # [h*N, 1, dk0]
        self.outputs = out

    def backward_auto(self, np_grads, np_key, np_w_k0, np_b0, np_indices):
        # [h*N, 1, dk0]
        grads = tf.placeholder(tf.float32,
                              shape=[
                                     self.head_num * self.bs_big,
                                     self.seq_len,
                                     self.dk0])
        g = tf.gradients(self.outputs, [self.key, self.w_k0, self.b0],
                         grad_ys=grads)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g_out = sess.run(g,
                             feed_dict={
                                 self.key: np_key,
                                 self.w_k0: np_w_k0,
                                 self.b0: np_b0,
                                 self.indices: np_indices,
                                 grads: np_grads
                             })
        d_input= g_out
        return d_input

    def backward_custom(self, np_grads, np_key, np_w_k0, np_b0):
        """
        grad: [M, T, O]
        """
        # [M, S, L]
        t_key = tf.constant(np_key, dtype=tf.float32)
        # [L, h*dk0]
        t_w_k0 = tf.constant(np_w_k0, dtype=tf.float32)
        # [h*dk0]
        # b0 = tf.constant(np_b0, dtype=tf.float32)
        # [h*N, S, dk0]
        grads = tf.constant(np_grads, dtype=tf.float32)

        #grads = tf.reshape(grads, shape=[self.head_num, self.bs_big, self.seq_len, self.dk0])
        #grads = tf.transpose(grads, perm=[1, 2, 0, 3])
        # [N, S, h*dk0]
        #grads = tf.reshape(grads, shape=[self.bs_big, self.seq_len, self.head_num * self.dk0])

        # [N, S, h*dk0] --> [M, S, h*dk0]

        # [M, S, L]
        
        #d_key = tf.matmul(grads, td_keyf.transpose(t_w_k0, perm=(1, 0)))
        d_key = tf.matmul(grads, tf.transpose(t_w_k0, perm=(1, 0)))
        print("### d_key shape ", d_key.shape)
        # [M, L, h*dk0]
        
        d_w = tf.matmul(tf.transpose(t_key, perm=(0, 2, 1)), grads)
        print("### 1 d_w shape ", d_w.shape)
        # [L, h*dk0]d_w
        d_w = tf.reduce_sum(d_w, axis=0)
        print("### 2 d_w shape ", d_w.shape)
        #d_inter = tf.identity(d_w)
        d_w = tf.identity(d_w)

        # [M, S, h*dk0]
        d_b = tf.reduce_sum(grads, axis=0)
        # [h*dk0]
        d_b = tf.reduce_sum(d_b, axis=0)
        print("### d_b shape ", d_b.shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            d_key, d_w, d_b = sess.run([d_key, d_w, d_b])
        return d_key, d_w, d_b

    def tf_custom_op(self, np_grads, np_key, np_w_k0, np_b0, np_indices):
        # [M, S, L]
        t_key = tf.constant(np_key, dtype=tf.float32)
        # [L, h*dk0]
        t_w_k0 = tf.constant(np_w_k0, dtype=tf.float32)
        # [h*dk0]
        t_b0 = tf.constant(np_b0, dtype=tf.float32)
        # [h*N, 1, dk0]
        t_grads = tf.constant(np_grads, dtype=tf.float32)
        # [N]
        t_indices = tf.constant(np_indices, dtype=tf.int32)

        # load so
        custom_module_lib_file = "/local/whale_kernel/build/src/libcustomop.so"
        custom_module = tf.load_op_library(custom_module_lib_file)

        d_key, d_w, d_b = custom_module.feed_forward_grad(
            grad_key=t_grads,
            key=t_key,
            w0=t_w_k0,
            b0=t_b0,
            indices=t_indices,
            head_num=self.head_num,
            #hidden_size=self.hidden_size,
            D_Q_0=self.dq0,
            D_Q_1=self.dq1,
            D_K_0=self.dk0
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            my_dk, my_dw, my_db = sess.run([d_key, d_w, d_b])

        print("### my_dk shape ", my_dk.shape)

        print("### my_dw shape ", my_dw.shape)
        print("### my_db shape ", my_db.shape)
        return my_dk, my_dw, my_db

def main():
    np_key = np.random.normal(size=(M, S, L))
    np_key = np_key.astype(np.float32)

    # np_grads = np.random.normal(size=(h*N, S, DQ1))
    # np_grads = np_grads.astype(np.float32)
    np_grads = np.ones((h*N, S, DQ1), np.float32)

    np_w_k0 = np.random.normal(size=(L, h*DQ1))
    np_w_k0 = np_w_k0.astype(np.float32)

    np_b0 = np.random.normal(size=(h*DQ1))
    np_b0 = np_b0.astype(np.float32)

    np_indices = np.random.randint(0, M, size=N, dtype=np.int32)
    # np_indices = np.array([0,1,0,1,0], dtype=np.int)

    model = test_backward(M, N, S, h, DK0, DK0, DQ1, L)
    print("### np_grads.shape ", np_grads.shape)
    print("### np_key.shape ", np_key.shape)
    print("### np_w_k0.shape ", np_w_k0.shape)
    print("### np_b0.shape ", np_b0.shape)

    [ref_d_key, ref_dw, ref_db] = model.backward_auto(np_grads, np_key, np_w_k0, np_b0, np_indices)
    print("### Peak ref dk shape = ", ref_d_key.shape)
    print("### Peak ref dw shape = ", ref_dw.shape)
    print("### Peak ref db shape = ", ref_db.shape)
    np.savetxt("ref_dk.txt", ref_d_key.reshape(-1))
    np.savetxt("ref_dw.txt", ref_dw.reshape(-1))
    np.savetxt("ref_db.txt", ref_db.reshape(-1))

    
    [tf_d_key, tf_dw, tf_db] = model.backward_custom(np_grads, np_key, np_w_k0, np_b0)
    #print(tf_d_key.shape)
    np.savetxt("tf_dk.txt", tf_d_key.reshape(-1))
    np.savetxt("tf_dw.txt", tf_dw.reshape(-1))
    np.savetxt("tf_db.txt", tf_db.reshape(-1))
    #tf_inter = tf_inter.reshape(-1)
    #np.savetxt("tf_output.txt", tf_inter)

    my_dk, my_dw, my_db = model.tf_custom_op(np_grads, np_key, np_w_k0, np_b0, np_indices)
    #my_dk = tf.transpose(my_d_key, perm=(0, 2, 1))
    np.savetxt("my_dk.txt", my_dk.reshape(-1))
    np.savetxt("my_dw.txt", my_dw.reshape(-1))
    np.savetxt("my_db.txt", my_db.reshape(-1))
    #my_inter = my_inter.reshape(-1)
    #np.savetxt("my_output.txt", my_inter)

    # np.savetxt("diff_output.txt", ref_d_key.reshape(-1) - my_d_key.reshape(-1))

    # print("validity: " + str(np.allclose(ref_d_key, my_d_key, atol = 1e-5)))
    # print("max diff " + str(np.fabs(ref_d_key - my_d_key).max()))
    # print("min diff " + str(np.fabs(ref_d_key - my_d_key).min()))

    # print("validity: " + str(np.allclose(ref_dw, my_dw, atol = 1e-5)))
    # print("max diff " + str(np.fabs(ref_dw - my_dw).max()))
    # print("min diff " + str(np.fabs(ref_dw - my_dw).min()))

    # print("validity: " + str(np.allclose(ref_db, my_db, atol = 1e-5)))
    # print("max diff " + str(np.fabs(ref_db - my_db).max()))
    # print("min diff " + str(np.fabs(ref_db - my_db).min()))

    # print("validity: " + str(np.allclose(ref_inter, my_inter, atol = 1e-5)))
    # print("max diff " + str(np.fabs(ref_inter - my_inter).max()))
    # print("min diff " + str(np.fabs(ref_inter - my_inter).min()))

if __name__ == "__main__":
    main()

