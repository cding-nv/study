#!/bin/sh

rm -rf ./temp *.o fmha_exe
mkdir temp
## Build fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu

nvcc   -std=c++17 -ccbin g++ -use_fast_math -Xptxas=-v --expt-relaxed-constexpr -g -lineinfo --keep --keep-dir ./temp -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION  -DUSE_I2F_EMULATION_TRICK -DUSE_F2I_EMULATION_TRICK -gencode=arch=compute_89,code=\"sm_89\" -I./ -I./generated -I/usr/local/cuda/include -c -o fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu.o generated/fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu


## Build softmax_fp16.cu

nvcc  -O3 -std=c++17 -ccbin g++ -use_fast_math -Xptxas=-v --expt-relaxed-constexpr -g  -lineinfo --keep --keep-dir ./temp -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION  -I./ -I./generated -I/usr/local/cuda/include -gencode=arch=compute_89,code=\"sm_89\" -c -o softmax_fp16.cu.o softmax_fp16.cu


## Build convert.cu
nvcc  -O3 -std=c++17 -ccbin g++ -use_fast_math -Xptxas=-v --expt-relaxed-constexpr -g  -lineinfo --keep --keep-dir ./temp -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION -I./ -I./generated -I/usr/local/cuda/include -gencode=arch=compute_89,code=\"sm_89\" -c -o convert.cu.o convert.cu

## Build fused_multihead_attention.cpp
g++  -O3 -std=c++17 -g -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION -I./ -I./generated -I/usr/local/cuda/include -I/usr/local/cuda/include -c -o fused_multihead_attention.cpp.o fused_multihead_attention.cpp

## Build fmha_exe
g++  -O3 -std=c++17 -g -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION -o fmha_exe fused_multihead_attention.cpp.o fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu.o convert.cu.o softmax_fp16.cu.o -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt



