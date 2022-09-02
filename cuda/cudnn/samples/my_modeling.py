# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# This file is mostly the same as bert/modeling.py from Google's BERT repository https://github.com/google-research/bert
# with configurable float types by setting tf.flags.FLAGS.floatx

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
# import copy
# import json
# import math
# import re
# import numpy as np
import six
import tensorflow as tf

def attention_layer(query, D_Q_0, D_Q_1, D_K_0,
                        head_num, is_fp16=False, scope='attr'):
    """
    Args:
        query(tf.tensor): (N, 1, L)
        key (tf.tensor):  (M, S, L), S = 50, 100, 150
        head_num: 3
        indices: (N)
    Returns:
        out: a tensor of shape (N, S, D_K_0)
    """
    assert D_Q_1 == D_K_0

    from_shape = get_shape_list(query, expected_rank=[3])
    #to_shape = get_shape_list(key, expected_rank=[3])
    assert from_shape[1] == 1
    # assert from_shape[0] == to_shape[0]
    batch_n = from_shape[0]
    #batch_m = to_shape[0]
    from_seq_len = 1
    #to_seq_len = to_shape[1]

    print('###from_shape =', from_shape[0], '/', from_shape[1], '/', from_shape[2])
    print('###D_Q_0 / head_num=', D_Q_0, '/', head_num)

    with tf.variable_scope(scope):
        #query
        # (N, 1, L)
        
        query = tf.layers.dense(query, D_Q_0 * head_num, use_bias=False, name='query_0') # (N, 1, D_Q_0 * head_num)
        # query = tf.layers.PRuLU(query)  # (N, 1, D_Q_0 * head_num)
        print('####Peak: dimensionality of query D_Q_0 * head_num = ', D_Q_0 * head_num)

        # (head_num * N, 1, D_Q_0)
        #query = tf.concat(tf.split(query, head_num, axis=2), axis=0)

        query_1 = tf.identity(query)

        # key
        #key = tf.layers.dense(key, D_K_0 * head_num, name='key') # (M, S, D_K_0 * head_num)
        # key = tf.layers.PReLU(key)   #(N, S, D_K_0 * head_num)
        #print('####Peak: dimensionality of key D_K_0 * head_num = ', D_K_0 * head_num)

        # [N, S, h*D_K_0]
        #key = tf.gather(key, indices, axis=0)

        # (head_num * N, S, D_K_0)
        #key = tf.concat(tf.split(key, head_num, axis=2), axis=0)
        #key_1 = tf.identity(key)

        #query = tf.layers.dense(query, D_Q_1, name='query_1') # (head_num * N, 1, D_Q_1)
        #query_2 = tf.identity(query)
        # query = tf.layers.PRuLU(query)  # (head_num * N, 1, D_Q_1)

        # matmul
        #qk = tf.matmul(key, query, transpose_b=True)  # (head_num * N, S, 1)

        #alphas = tf.nn.softmax(qk, axis=1)  # (head_num * N, S, 1)

        #inter_out = tf.identity(alphas, "inter_out")
        # inter_out = tf.reshape(inter_out, [head_num, batch_n, to_seq_len, from_seq_len])
        # inter_out = tf.transpose(inter_out, perm=[1, 0, 2, 3])
        # inter_out = tf.identity(key, "inter_out")
        # inter_out = tf.reshape(inter_out, [head_num, batch_n, to_seq_len, D_K_0])
        # inter_out = tf.transpose(inter_out, perm=[1, 0, 2, 3])

        # out = tf.multiply(key, alphas)  # (head_num * N, S, D_K_0)
        #out = tf.matmul(key, alphas, transpose_a=True) # [h*N, dk0, 1]

        # out = tf.reshape(out, [head_num, batch_n, D_K_0, 1])
        # out = tf.transpose(out, perm=[1, 0, 2, 3]) # [N, h, dk0, 1]

        # out = tf.split(out, head_num, axis=0)

    return query_1

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
