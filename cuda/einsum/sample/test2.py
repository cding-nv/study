# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import os
import sys
import tensorflow.contrib.eager as tfe

import nvtx.plugins.tf as nvtx_tf

import time, timeit

# input[0] (E,G,C,M) (10, 1, 8, 16)
E = 20
G = 1
C = 512
M = 1024

batch = E * C
len1 = 1
len2 = 1
len3 = M

# input[1] (E,M,H)  (10,16,18)
H = 9984

custom_module_lib_file =  "../build/src/tf_op/libcustomop.so"
custom_module = tf.load_op_library(custom_module_lib_file)

np_data = np.random.choice([1,1,1,1], batch)  # (batch,)
np_data = np_data.astype(np.float32)

np_data = np.expand_dims(np_data, axis=1)  # (batch, 1)
np_data = np.tile(np_data, (1, len1*len2*len3))
np_data = np.reshape(np_data, (batch, len1, len2, len3))

tf_data = tf.constant(np_data, dtype=tf.float32)

np_data1 = np.reshape(np_data, (E,G,C,M))
tf_data1 = tf.constant(np_data1, dtype=tf.float32)
np_data2 = np.full((E,M,H), 2.0)
tf_data2 = tf.constant(np_data2, dtype=tf.float32)

#tf_data2, nvtx_context = nvtx_tf.ops.start(tf_data2, message='orig_einsum')
res_orig = tf.einsum('EGCM, EMH->EGCH', tf_data1, tf_data2)
#res_orig = nvtx_tf.ops.end(res_orig, nvtx_context)

def check_result(input_data, indices):
    # check the result
    for i in range(batch):
        a = input_data[i, 0, 0, 0]
        if (a > 0):
            if (i != indices[i] - 1):
                print(a)
                print(i)
                print(indices[i])
                return False
        else:
            if (indices[i] != 0):
                print(a)
                print(i)
                print(indices[i])
                return False
    return True

# if check_result(np_data, np_indices):
#     print("Test Passed!")
# else:
#     print("Test Failed!")


output_indices = custom_module.non_zero_index_grad_gpu(tf_data)
#output_indices, nvtx_context = nvtx_tf.ops.start(output_indices, message='custom_einsum')

# Get valid index
e_group = tf.split(output_indices, E, axis=0)
index_group_v = tf.where(tf.reshape(e_group,(E,G,C)))

# Get non zero counter
nonzero_num = tf.math.count_nonzero(e_group, axis=1)

# Get valid data from data1
position4 = tf.reshape(index_group_v, (-1, 1, 1, 3))
tf_data1_v = tf.gather_nd(tf_data1, position4)

tf_data2_group = tf.split(tf_data2, E, axis=0)

# E group valid data1 einsum E group data2
acc = 0
res_v_list = []
for i in range(E):
    tf_data1_v_i = tf.gather(tf_data1_v, tf.range(acc, acc + nonzero_num[i]))
    tf_data1_v_i = tf.reshape(tf_data1_v_i, (1, nonzero_num[i], M))
    acc += nonzero_num[i]
    res_v_i = tf.einsum('ECM, EMH->ECH', tf_data1_v_i, tf_data2_group[i])
    res_v_i = tf.reshape(res_v_i, (-1, 1, H))
    res_v_list.append(res_v_i)
res_v = tf.concat(res_v_list, axis=0)

res_v = tf.reshape(res_v, (-1, 1, 1, H))

# Get position of every valid data in real result.
position_v_pad = index_group_v[:, 0:1] * 10 + index_group_v[:, 2:3]
position_v_pad = tf.cast(position_v_pad, tf.int32)
v_shape = tf.constant([E*G*C, 1, 1, H])
res_v_pad = tf.scatter_nd(position_v_pad, res_v, v_shape)
res_v_pad = tf.reshape(res_v_pad, (E,G,C,H))
#res_v_pad = nvtx_tf.ops.end(res_v_pad, nvtx_context)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_data1 = sess.run(tf_data1)
    tf_data2 = sess.run(tf_data2)
    tic0 = timeit.default_timer()
    data2 = sess.run(res_orig)
    toc0 = timeit.default_timer()
    print("time for the ref TF: %.4f ms"%((toc0-tic0)*1000))

    #output_indices=sess.run(output_indices)
    #e_group = sess.run(e_group)
    #print(e_group)   
    
    #nonzero_num = sess.run(nonzero_num)
    #print("nonzero_num ", nonzero_num)
    
    #t3 = timeit.default_timer()
    #index_group_v = sess.run(index_group_v)
    #print("index_group_v: ", index_group_v)

    #t2 = timeit.default_timer()
    #tf_data1_v = sess.run(tf_data1_v)
    #print("tf_data1_v shape ", tf_data1_v.shape)
    #print(tf_data1_v)

    #tic1 = timeit.default_timer()
    tbc1 = timeit.default_timer()
    res_v = sess.run(res_v)
    print("res_v shape ", res_v.shape)
    #print(res_v)

    #position_v_pad = sess.run(position_v_pad)
    #print("position_v_pad: ", position_v_pad)

    #res_v_pad = sess.run(res_v_pad)
    #print("res_v_pad shape ", res_v_pad.shape)

    #for i in range(res_v_pad.shape[0]):
        #print(res_v_pad[i])

#exit()

toc1 = timeit.default_timer()
#print("time for the ref TF: %.4f ms"%((t2-t3)*1000))
#print("time for the ref TF: %.4f ms"%((tic1-t2)*1000))
#print("time for the ref TF: %.4f ms"%((toc1-tic1)*1000))
print("time for the custom: %.4f ms"%((toc1-tbc1)*1000))
