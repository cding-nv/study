# 
# Copyright (c) 2021, 
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

project(FasterTransformer LANGUAGES CXX CUDA)

find_package(CUDA 10.1 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

#set(TF_PATH "/usr/local/lib/python3.6/dist-packages/tensorflow_core" CACHE STRING "TensorFlow path")
set(TF_PATH "/home/pai/lib/python3.6/site-packages/tensorflow_core" CACHE STRING "TensorFlow path")

if(NOT TF_PATH)
  message(FATAL_ERROR "TF_PATH must be set.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for.")

if("${GPU_ARCHS}" STREQUAL "")
	set(GPU_ARCHS "70")
endif()

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch} -rdc=true")
	if(SM STREQUAL 70 OR SM STREQUAL 75)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
	endif()
endforeach()

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++14")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3 --use_fast_math")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

list(APPEND COMMON_HEADER_DIRS ${TF_PATH}/include)
list(APPEND COMMON_LIB_DIRS ${TF_PATH})

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(src)

if(NOT DEFINED LIB_INSTALL_PATH)
	set(LIB_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR})
endif()

#install(TARGETS customop
#	LIBRARY DESTINATION ${LIB_INSTALL_PATH}
#)

