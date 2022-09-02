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

np_data = np.random.choice([0,1,2,3], batch)  # (batch,)
np_data = np_data.astype(np.float32)

np_data = np.expand_dims(np_data, axis=1)  # (batch, 1)
np_data = np.tile(np_data, (1, len1*len2*len3))
np_data = np.reshape(np_data, (batch, len1, len2, len3))

tf_data = tf.constant(np_data, dtype=tf.float32)

np_data1 = np.reshape(np_data, (E,G,C,M))
tf_data1 = tf.constant(np_data1, dtype=tf.float32)
np_data2 = np.full((E,M,H), 2.0)
tf_data2 = tf.constant(np_data2, dtype=tf.float32)

tf_data2, nvtx_context = nvtx_tf.ops.start(tf_data2, message='orig_einsum')

res_orig = tf.einsum('EGCM, EMH->EGCH', tf_data1, tf_data2)
res_orig = nvtx_tf.ops.end(res_orig, nvtx_context)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_data1 = sess.run(tf_data1)
    tf_data2 = sess.run(tf_data2)
    tic0 = timeit.default_timer()
    data2 = sess.run(res_orig)
    toc0 = timeit.default_timer()
    print("time for the ref TF: %.4f ms"%((toc0-tic0)*1000))
