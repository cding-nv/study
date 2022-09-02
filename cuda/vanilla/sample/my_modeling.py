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
import math
# import re
# import numpy as np
import six
import tensorflow as tf


def dropout(input_tensor, dropout_prob, seed=None):
  """Perform dropout.
  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).
  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob, seed=seed)
  return output

def attention_layer_custom(queries,
                    keys,
                    values,
                    key_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    is_fp16=False,
                    do_lrelu=False,
                    lrelu_alpha=1,
                    attention_probs_dropout_prob=0.2
                         ):

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(queries, expected_rank=[2, 3])
  to_shape = get_shape_list(keys, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`
  if is_fp16:
      FT_TYPE = tf.float16
  else:
      FT_TYPE = tf.float32


  def my_leaky_relu(x):
      return tf.nn.leaky_relu(x, alpha=lrelu_alpha)

  query_act=my_leaky_relu
  key_act=my_leaky_relu
  value_act=my_leaky_relu

  from_tensor_2d = reshape_to_matrix(queries)
  to_tensor_2d = reshape_to_matrix(keys)
  v_tensor_2d = reshape_to_matrix(values)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="my_query",
      kernel_initializer=tf.glorot_uniform_initializer(0, dtype=FT_TYPE))
  # `key_layer` = [B*T, N*H]

  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="my_key",
      kernel_initializer=tf.glorot_uniform_initializer(0, dtype=FT_TYPE))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      v_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="my_value",
      kernel_initializer=tf.glorot_uniform_initializer(0, dtype=FT_TYPE))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

  inter_out = tf.identity(attention_scores)

  random_tensor = tf.random_uniform(shape=[batch_size, num_attention_heads, from_seq_length, to_seq_length], seed=0, dtype=tf.float32)
  keep_mask = random_tensor >= attention_probs_dropout_prob
  custom_module_lib_file = "../build/src/libcustomop.so"
  custom_module = tf.load_op_library(custom_module_lib_file)
  attention_probs, _ = custom_module.multi_head_attention(
      from_tensor=attention_scores,
      k_mask=key_mask,
      keep_mask=keep_mask,
      batch=batch_size,
      head_num = num_attention_heads,
      size_per_head=size_per_head,
      dropout_rate=attention_probs_dropout_prob
  )

###  attention_scores = tf.multiply(attention_scores,
###                                 1.0 / math.sqrt(float(size_per_head)))
###
###  if key_mask is not None:
###    # `key_mask` = [B, 1, T]
###    key_mask = tf.expand_dims(key_mask, axis=[1])
###    # [B, 1, 1, T]
###    key_mask = tf.expand_dims(key_mask, axis=[1])
###    key_mask = tf.tile(key_mask, [1, num_attention_heads, from_seq_length, 1])
###
###    # Since key_mask is 1.0 for positions we want to attend and 0.0 for
###    # masked positions, this operation will create a tensor which is 0.0 for
###    # positions we want to attend and -10000.0 for masked positions.
###    adder = (1.0 - tf.cast(key_mask, attention_scores.dtype)) * -10000.0
###
###    # Since we are adding it to the raw scores before the softmax, this is
###    # effectively the same as removing these entirely.
###    attention_scores += adder
###
###  # Normalize the attention scores to probabilities.
###  # `attention_probs` = [B, N, F, T]
###  attention_probs = tf.nn.softmax(attention_scores)
###
###  # This is actually dropping out entire tokens to attend to, which might
###  # seem a bit unusual, but is taken from the original Transformer paper.
###  # attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
###
  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if False:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer, inter_out


def attention_layer_orig(queries,
                    keys,
                    values,
                    key_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    is_fp16=False,
                    do_lrelu=False,
                    lrelu_alpha=1,
                    attention_probs_dropout_prob=0.2
                         ):

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(queries, expected_rank=[2, 3])
  to_shape = get_shape_list(keys, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`
  if is_fp16:
      FT_TYPE = tf.float16
  else:
      FT_TYPE = tf.float32


  def my_leaky_relu(x):
      return tf.nn.leaky_relu(x, alpha=lrelu_alpha)

  query_act=my_leaky_relu
  key_act=my_leaky_relu
  value_act=my_leaky_relu

  from_tensor_2d = reshape_to_matrix(queries)
  to_tensor_2d = reshape_to_matrix(keys)
  v_tensor_2d = reshape_to_matrix(values)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=tf.glorot_uniform_initializer(0, dtype=FT_TYPE))
  # `key_layer` = [B*T, N*H]

  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=tf.glorot_uniform_initializer(0, dtype=FT_TYPE))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      v_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=tf.glorot_uniform_initializer(0, dtype=FT_TYPE))

  # print("############# batch_size: " + str(batch_size))
  # print("############# num_attention_heads: " + str(num_attention_heads))
  # print("############# from_seq_length: " + str(from_seq_length))
  # print("############# size_per_head: " + str(size_per_head))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  inter_out = tf.identity(attention_scores)

  # attention_scores = tf.multiply(attention_scores,
                                 # 1.0 / math.sqrt(float(size_per_head)))
  attention_scores = attention_scores / (size_per_head  ** 0.5)

  if key_mask is not None:
    # `key_mask` = [B, 1, T]
    key_mask = tf.expand_dims(key_mask, axis=[1])
    # [B, 1, 1, T]
    key_mask = tf.expand_dims(key_mask, axis=[1])
    key_mask = tf.tile(key_mask, [1, num_attention_heads, from_seq_length, 1])

    # Since key_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(key_mask, attention_scores.dtype)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob, seed=0)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if False:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer, inter_out

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


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


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
