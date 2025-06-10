cmake_minimum_required(VERSION 3.16)

#set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}   -D_ENABLE_FLASH_ATTENTION")
#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}   -D_ENABLE_FLASH_ATTENTION")

add_executable(llama2_example llama2_example.cpp)
target_link_libraries(llama2_example PUBLIC sycl_kernels xetla_kernels_fmha)

