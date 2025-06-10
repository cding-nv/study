# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

cmake_minimum_required(VERSION 3.16)

#add_library(LlamaWeight STATIC LlamaWeight.cc)
#set_property(TARGET LlamaWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
#set_property(TARGET LlamaWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
#target_link_libraries(LlamaWeight PUBLIC LlamaDecoderLayerWeight cuda_utils logger)

#add_library(Llama STATIC Llama.cc)
#set_property(TARGET Llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
#set_property(TARGET Llama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
#target_link_libraries(Llama PUBLIC LlamaWeight)

add_library(llama_sycl STATIC llama2_sycl.cc)

#set(SYCL_KERNEL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/xfastertransformer/kernels/kernels")
#target_include_directories(llama_sycl PUBLIC ${SYCL_KERNEL_INCLUDE_DIRS} )

set_property(TARGET llama_sycl PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(llama_sycl PRIVATE "${TORCH_LIBRARIES}" sycl_kernels xetla_kernels_fmha)

#set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}   -D_ENABLE_FLASH_ATTENTION")
#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -D_ENABLE_FLASH_ATTENTION")

