/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


//We can disable the FADD trick for archs with F2IP
#if 0 // disable_fadd_trick
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif // USE_I2F_EMULATION_TRICK

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif // USE_F2I_EMULATION_TRICK
#endif // disable_fadd_trick

#include <cuda.h>

#if CUDA_VERSION >= 11000

#include <fused_multihead_flash_attention_kernel_noloop.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>
#include <fused_multihead_flash_attention_kernel.h>




using Launch_params = bert::Fused_multihead_attention_launch_params;
using Attention_mask_type = fmha::Attention_mask_type;

#if 0 // has_noloop (unconditionally disabled since not maintained & not actively used)
using Kernel_traits = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u>;

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention<Kernel_traits>(params);
}

void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){

  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( smem_size >= 48*1024 ) {
    FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
  }
  dim3 grid(params.h, params.b);
  fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(params);
}

#endif // has_noloop

#if 1 && !1 // has_noloop && !tiled
using Kernel_traits_nl = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*dense mask*/ 2,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t,
    0,
    0,
    0>;

using Kernel_traits_nl_causal = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*causal mask*/ 3,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t>;

using Kernel_traits_nl_sliding_or_chunked_causal = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*sliding window causal mask*/ 4,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t>;

using Kernel_traits_nl_custom_mask = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*custom mask*/ 5,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t>;

#if 1 // padding_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl>(params);
}

#endif // padding mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_causal>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_sliding_or_chunked_causal>(params);
}

#endif // sliding_or_chunked_causal_mask

#if 1 // custom_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_custom_mask>(params);
}

#endif // custom_mask

void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_nl(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){

  // runtime q_loop_iters
  int loop_iters = ( params.s + 128 - 1 )  / 128;
  // dim3 grid(params.h, params.b, loop_iters);
  dim3 grid(loop_iters, params.h, params.b); // better locality
  constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
#endif // causal mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 1 // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
#endif // sliding_or_chunked_causal_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {
#if 1 // padding_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
#endif // padding_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {
#if 1 // custom_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
#endif // custom mask
  }
}

#endif // has_noloop && !tiled

#if 1 // tiled

using Kernel_traits_nl_tiled = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*dense mask*/ 2,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t,
    0,
    0,
    0>;

using Kernel_traits_nl_tiled_causal = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*causal mask*/ 3,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t>;

using Kernel_traits_nl_tiled_sliding_or_chunked_causal = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*sliding window causal mask*/ 4,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t>;

using Kernel_traits_nl_tiled_custom_mask = fmha::Kernel_traits_v2<
    fmha::Ampere_hmma_fp16_traits,
    128,
    16,
    0,
    128,
    4,
    1,
    1,
    0x1007u | 0x200 /* no_loop flag */,
    /*custom mask*/ 5,
    /*bmm2_fp16_epilogue*/ true,
    fmha::fp16_t>;

#if 1 // padding_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled>(params);
}

#endif // padding_mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_causal>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_sliding_or_chunked_causal>(params);
}

#endif // sliding_or_chunked_causal_mask

#if 1 // custom_mask

extern "C"
__global__
void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_custom_mask>(params);
}

#endif // custom mask

// Granular tiling
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_nl_tiled(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){
  // runtime q_loop_iters
  using Cta_tile_o = typename Kernel_traits_nl_tiled::Cta_tile_o;
  int ctas_per_o_row = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
  int loop_iters = ( params.s + 128 - 1 )  / 128;
  dim3 grid(loop_iters * ctas_per_o_row, params.h, params.b);
  constexpr int smem_size = Kernel_traits_nl_tiled::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
#endif // causal mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 1 // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl_tiled,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
#endif // sliding_or_chunked_causal_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {
#if 1 // padding_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
#endif // padding_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {
#if 1 // custom_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(params);
#endif // custom mask
  }
}

#endif // tiled

#else // CUDA_VERSION >= 11000

void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_nl(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_nl_tiled(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

#endif // CUDA_VERSION >= 11000

