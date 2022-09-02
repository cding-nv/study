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

np_data = np.random.choice([0,1,0,0,0,0], batch)  # (batch,)
np_data = np_data.astype(np.float32)

np_data = np.expand_dims(np_data, axis=1)  # (batch, 1)
np_data = np.tile(np_data, (1, len1*len2*len3))
np_data = np.reshape(np_data, (batch, len1, len2, len3))

tf_data = tf.constant(np_data, dtype=tf.float32)

np_data1 = np.reshape(np_data, (E,G,C,M))
tf_data1 = tf.constant(np_data1, dtype=tf.float32)
np_data2 = np.full((E,M,H), 2.0)
tf_data2 = tf.constant(np_data2, dtype=tf.float32)
res_orig = tf.einsum('EGCM, EMH->EGCH', tf_data1, tf_data2)

tbc1 = timeit.default_timer()
output_indices = custom_module.non_zero_index_grad_gpu(tf_data)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    np_indices=sess.run(output_indices)
    #print(np_indices)
    #tf_data1 = sess.run(tf_data1)
    #tf_data2 = sess.run(tf_data2)
    tic0 = timeit.default_timer()
    data2 = sess.run(res_orig)
    toc0 = timeit.default_timer()
    print("time for the ref TF: %.4f ms"%((toc0-tic0)*1000))
    exit()
    #print(data2.shape)
    #for i in range(data2.shape[0]):
        #print(data2[i])

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

if check_result(np_data, np_indices):
    print("Test Passed!")
else:
    print("Test Failed!")

e_group = np.split(np_indices, E, axis=0)

# Get max valid num in C axis 
def max_valid_num(group):
    # output_indices shape is (E, G, C)
    #output_indices = np.reshape(output_indices, [-1])
    #print(e_group)
    max_zero_counter = 0
    non_zero = np.count_nonzero(group, axis=1)
    #print("non_zero: ", non_zero)
    max_num_ = np.max(non_zero)
    print(max_num_)
    return max_num_

max_num = max_valid_num(e_group)

# Get non zero position in (E,G,C)
def nonzero_position(group):
    group = np.array(group)
    group = np.reshape(group, (E,G,C))
    position_ = np.argwhere(group > 0)
    # position shape is (x, 3), for example [9, 0, 6]. It's index of valid line
    #print(position_)
    print("position shape ", position_.shape)
    return position_

position = nonzero_position(e_group)

# reshape to (-1,1,1,3)
position4 = np.reshape(position, (-1, 1, 1, 3))
#print(position4)

np_data = np.reshape(np_data, (E, G, C, M))

tf_data = tf.convert_to_tensor(np_data)

# position_test = np.array([
#     [[[0,0,0],[0,0,1]]],
#     [[[1,0,2],[1,0,0]]]
# ])

tf_data_v = tf.gather_nd(tf_data, position4)
with tf.Session() as sess:
    #print("tf_data:")
    #print(sess.run(tf_data))
    #print("tf_data_v:")
    tf_data_v = sess.run(tf_data_v)
    #tf_data_v = tf_data_v.eval()
    #print(tf_data_v)
    print("tf_data_v shape: ", tf_data_v.shape)

# tf_data_v shape (x, 1, 1, 32)
# position shape (x, 3), for example [9, 0, 6]
index = 0
row0 = np.array([[[[0]*M]]])

for i in range(E):
    for j in range(max_num):
        if index < position.shape[0] and position[index][0] == i:
            #print("i ", i, " j", j, " index ", index)
            index += 1
        else:
            #print("i ", i, " j", j)
            tf_data_v = np.insert(tf_data_v, i * max_num + j, row0, axis = 0)
            #print("#### row0 shape ", row0.shape)
            #print("### tf_data_v shape ", tf_data_v.shape)
print("### tf_data_v shape ", tf_data_v.shape, " shape0 ", tf_data_v.shape[0])
#print(tf_data_v)

#for i in range(tf_data_v.shape[0]):
#    print(tf_data_v[i])

data_v = np.reshape(tf_data_v, (E, G, max_num, M))
#print("data_v shape :", data_v.shape)
#print(data_v)

tf_data1_v = tf.constant(data_v, dtype=tf.float32)
res_v = tf.einsum('EGCM, EMH->EGCH', tf_data1_v, tf_data2)

# Get valid data position from padding result res_v
print(position.shape)
index_0 = 0
pos = 0
position_v = np.arange(position.shape[0])
for i in range(E * max_num):
    if (pos >= position.shape[0]):
        break
    if position[pos][0] == index_0:
        position_v[pos] = i
        pos += 1
    if (i + 1) % max_num == 0:
        index_0 += 1
#print(position_v)
#position_v = np.reshape(position_v, (1, position.shape[0]))

position_v_pad = np.arange(position.shape[0])
for i in range(position.shape[0]):
    position_v_pad[i] = position[i][0] * C + position[i][2]

res_v = tf.reshape(res_v, [E*G*max_num, 1, 1, H])
tic1 = timeit.default_timer()
## Begin to valid einsum
with tf.Session() as sess:
    res_v = sess.run(res_v)
    #print(res_v)
    res_v = res_v[position_v]
    print("rev_v shape ", res_v.shape)
    #print(res_v)
    #print(position_v)
    v_indices = tf.constant(position_v_pad, dtype=tf.int32)
    v_indices = tf.reshape(v_indices, [position_v_pad.shape[0], 1])
    v_shape = tf.constant([E*G*C, 1, 1, H])
    res_v_pad = tf.scatter_nd(v_indices, res_v, v_shape)
    res_v_pad = tf.reshape(res_v_pad, (E,G,C,H))
    res_v_pad = sess.run(res_v_pad)
    #print(res_v_pad.shape)

toc1 = timeit.default_timer()
print("time for the ref TF: %.4f ms"%((toc1-tic1)*1000))
print("time for the ref TF: %.4f ms"%((toc1-tbc1)*1000))
    
#for i in range(res_v_pad.shape[0]):
    #print(res_v_pad[i])