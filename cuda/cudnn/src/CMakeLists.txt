# 
# Copyright (c) 2019, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)

set(customop_files
	feed_forward_op.cpp
	transformer/cublas_wrappers.cu
	feed_forward_grad_op.cpp
	transformer/general_kernels.cu
	softmax_op.cpp
	transformer/softmax_kernels.cu
	softmax_grad_op.cpp
        dropout_op.cpp
        transformer/dropout_kernels.cu
        layernorm_op.cpp
        transformer/normalize_kernels.cu
        multi_head_attention_op.cpp
        multi_head_attention_grad_op.cpp
        mha/multiHeadAttention.cpp
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DNDEBUG)

include_directories(
  includes
  mha
)

add_library(customop SHARED
  ${customop_files}
)
set_target_properties(customop PROPERTIES
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(customop PRIVATE
  -lcublas
  -lcudart
  -lcudnn
  -ltensorflow_framework
  ${CMAKE_THREAD_LIBS_INIT}
)

