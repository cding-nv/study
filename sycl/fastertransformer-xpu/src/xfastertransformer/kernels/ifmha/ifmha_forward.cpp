#include "ifmha_forward.h"

namespace gpu::xetla {
namespace ifmha {

template <
    typename ifmha_policy,
    typename T,
    bool kUseBias,
    bool kIsTraining>
class IfmhaForwardKernel;

// The launcher of indexed flash mha forward kernel
template <typename ifmha_policy, typename T, bool kUseAlibi, bool kUseBias,
          bool kIsTraining>
sycl::event
ifmha_forward_impl(sycl::queue &q, T *query, T *key0, T *key1, T *value0,
                   T *value1, int32_t *index, T* alibi, T *bias, uint8_t *dropout,
                   float dropout_prob, float sm_scale, T *out,
                   uint32_t num_batches, uint32_t num_heads, uint32_t head_size,
                   uint32_t kv_len0, uint32_t kv_len1, uint32_t padded_kvlen) {
  // ifmha forward kernel
  using ifmha_forward_op_t =
      ifmha_forward_t<ifmha_policy, T, kUseAlibi, kUseBias, kIsTraining>;

  sycl::nd_range<2> NdRange = ifmha_forward_op_t::get_nd_range(
      num_batches, ifmha_policy::Beams, num_heads);

  auto event = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(  //        class IfmhaForwardKernel<ifmha_policy, T, kUseBias, kIsTraining>
        NdRange, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
      // exec item
      xetla_exec_item<2> ei(item);

      // init ifmha forward op and arguments
      ifmha_forward_op_t ifmha_fwd_op;
      typename ifmha_forward_op_t::arguments_t
          args(query, key0, key1, value0, value1, index, alibi, bias, dropout,
               dropout_prob, sm_scale, out, num_batches, num_heads, head_size,
               kv_len0, kv_len1, padded_kvlen);

      // call the functor
      ifmha_fwd_op(ei, args);
        });
  });
  return event;
}

} // namespace ifmha

#define CALL_IMPL_FUNC(P)                                           \
  ifmha::ifmha_forward_impl<P, T, kUseAlibi, kUseBias, kIsTraining>( \
      q,                                                            \
      query,                                                        \
      key0,                                                         \
      key1,                                                         \
      value0,                                                       \
      value1,                                                       \
      index,                                                        \
      alibi,                                                        \
      bias,                                                         \
      dropout,                                                      \
      dropout_prob,                                                 \
      sm_scale,                                                     \
      out,                                                          \
      num_batches,                                                  \                                                     
      num_heads,                                                    \
      head_size,                                                    \
      kv_len0,                                                      \
      kv_len1,                                                      \
      padded_kvlen);
     // alibi_padding,                                               
     // attn_mask_padding)

/// @brief Main execution function for indexed flash mha forward.
template <typename T, bool kUseAlibi = false, bool kUseBias = false, bool kIsTraining = false>
sycl::event ifmha_forward(
    sycl::queue& q,
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* alibi,
    T* bias,
    uint8_t* dropout,
    float dropout_prob,
    float sm_scale,
    T* out,
    uint32_t num_batches,
    uint32_t beam,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t kv_len0,
    uint32_t kv_len1,
    uint32_t padded_kvlen) {
  if (head_size <= 64) {
    CALL_IMPL_FUNC(ifmha_policy_64x64);
  } else if (head_size <= 128) {
    CALL_IMPL_FUNC(ifmha_policy_256x128);
  } else if (head_size <= 256) {
    CALL_IMPL_FUNC(ifmha_policy_512x256);
  } else {
    std::cout << "No policy available for current head_size " << head_size
              << "\n";
    sycl::event evt;
    return evt;
  }
}

#undef CALL_IMPL_FUNC

void fmha_forward_index_kernel(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* key_cache,
    void* value_cache,
    int32_t* index,
    void* alibi,
    void* attn_mask,
    uint8_t* dropout,
    void* out,
    uint32_t timestep,
    float alpha,
    float beta,
    float dropout_p,
    uint32_t num_batches,
    uint32_t beam_width,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_queries,
    uint32_t num_keys_in,
    uint32_t num_keys_out,
    uint32_t padded_kvlen) {
    //uint32_t alibi_padding,
    //uint32_t attn_mask_padding,
    //bool is_causal) {
  using T = sycl::half;
  assert(num_queries == 1);
  //TORCH_CHECK(
  //    num_queries == 1,
  //    "SDP Index fusion kernel requires num_queries == 1 so far ...");
  //TORCH_CHECK(
  //    is_causal == false,
  //    "SDP Index fusion kernel doesn't support causal so far ...");

#define DISPATCH_TEMPLATE(T, USE_ALIBI, USE_BIAS, IS_TRAINING) \
  ifmha_forward<T, USE_ALIBI, USE_BIAS, IS_TRAINING>(          \
      q,                                                       \
      (T*)query,                                               \
      (T*)key,                                                 \
      (T*)key_cache,                                           \
      (T*)value,                                               \
      (T*)value_cache,                                         \
      index,                                                   \
      (T*)alibi,                                               \
      (T*)attn_mask,                                           \
      dropout,                                                 \
      dropout_p,                                               \
      alpha,                                                   \
      (T*)out,                                                 \
      num_batches,                                             \
      beam_width,                                              \
      num_heads,                                               \
      head_dim,                                                \
      num_keys_in,                                             \
      num_keys_out,                                            \
      padded_kvlen);
      //alibi_padding,                                          
      //attn_mask_padding);

  if (alibi) {
    if (attn_mask) {
      DISPATCH_TEMPLATE(T, true, true, false)
    } else {
      DISPATCH_TEMPLATE(T, true, false, false)
    }
  } else {
    if (attn_mask) {
      DISPATCH_TEMPLATE(T, false, true, false)
    } else {
      DISPATCH_TEMPLATE(T, false, false, false)
    }
  }
}
} // namespace gpu::xetla
