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
	multi_head_attention_op.cc
    multi_head_attention_op.cu.cc
    multi_head_attention.cu
    general_kernels.cu
    normalize_kernels.cu
    softmax_grad_op.cc
    softmax_grad_op.cu.cc
    softmax_grad.cu
    layernorm_grad_op.cc
    layernorm_grad_op.cu.cc
    layernorm_grad.cu
    dense_grad_op.cc
    dense_grad_op.cu.cc
    dense_grad.cu
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DNDEBUG)

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

