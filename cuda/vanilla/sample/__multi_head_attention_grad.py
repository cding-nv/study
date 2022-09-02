
#!/usr/bin/env python3
"""
Gradients for multi-head-attention.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import six


@ops.RegisterGradient("MultiHeadAttention")
def _multi_head_attention_grad_cc(op, grad, grad1=None, grad2=None):
    # score_tensor = op.inputs[0]
    keep_mask = op.inputs[2]

    # [h, N, T_q, T_k]
    softmax = op.outputs[1]

    batch_size = op.get_attr("batch")
    head_num = op.get_attr("head_num")
    size_per_head = op.get_attr("size_per_head")
    rate = op.get_attr("dropout_rate")

    #print("#################################")
    #print("#################################")
    #print("#################################")
    #print("#################################")
    #print("rate = " + str(rate))

    # score_shape = get_shape_list(q_tensor, expected_rank=4)
    # softmax_shape = get_shape_list(k_tensor, expected_rank=4)

    custom_lib = "../build/src/libcustomop.so"
    custom_lib = tf.load_op_library(custom_lib)

    # TODO: whether attention_probs is necessrary here???
    d_score = custom_lib.softmax_grad(
        grads=grad,
        softmax=softmax,
        dropout_mask=keep_mask,
        head_num=head_num,
        batch=batch_size,
        size_per_head=size_per_head,
        dropout_rate=rate
    )

    return d_score, None, None


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
