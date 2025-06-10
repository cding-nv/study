/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <memory>

#include "src/xfastertransformer/models/llama/LlamaWeight.h"

namespace fastertransformer {

enum class AttentionType {
    UNFUSED_MHA,
    UNFUSED_PADDED_MHA,
    FUSED_MHA,
    FUSED_PADDED_MHA
};

template<typename T>
class Llama {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;
    size_t rotary_embedding_dim_;
    float layernorm_eps_;

    static constexpr bool  neox_rotary_style_ = true;

    int    start_id_;
    int    end_id_;
    size_t hidden_units_;

    size_t    local_head_num_;

    AttentionType attention_type_;

    size_t     vocab_size_padded_;

    // Residual Type
    const bool use_gptj_residual_ = false;

    // Prompt Learning Parameters
    //PromptLearningType prompt_learning_type_;
    int                prompt_learning_start_id_;  // start_id for prompt_learning (only needed by prefix prompts)
    bool               has_prefix_prompt_;
    bool               has_prefix_soft_prompt_;

    void allocateBuffer(size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_cache_seq_len, size_t max_input_len);
    void freeBuffer();

    void initialize();

protected:
    T*       padded_embedding_kernel_;
    T*       padded_embedding_bias_;
    const T* padded_embedding_kernel_ptr_;

    T* input_attention_mask_;

    T* decoder_input_buf_;
    T* decoder_output_buf_;
    T* normed_decoder_output_buf_;

    float* logits_buf_;
    float* nccl_logits_buf_;
    float* cum_log_probs_;

    bool*     finished_buf_;
    bool*     h_finished_buf_;
    int*      sequence_lengths_          = nullptr;
    int*      tiled_total_padding_count_ = nullptr;
    uint32_t* seq_limit_len_             = nullptr;

    T*   key_cache_;
    T*   value_cache_;
    int* cache_indirections_[2] = {nullptr, nullptr};

    // prompt_learning weight_batch ptrs
    const T** prompt_learning_weight_batch_;
    int*      tiled_prompt_lengths_buf_;  // only needed by prefix prompts

    int*  tiled_input_ids_buf_;
    int*  tiled_input_lengths_buf_;
    int*  transposed_output_ids_buf_;
    int*  output_ids_buf_;
    int*  parent_ids_buf_;
    int*  start_ids_buf_;
    int*  end_ids_buf_;
    bool* masked_tokens_ = nullptr;

    bool* generation_should_stop_ = nullptr;

    T*     context_decoder_input_buf_;
    T*     context_decoder_output_buf_;
    float* output_log_probs_buf_;

    void*         token_generated_ctx_ = nullptr;

    // callback step
    size_t token_generated_cb_step_ = 5; // default 5, override by env LLAMA_STREAM_CB_STEP

public:
    Llama(size_t                              head_num,
          size_t                              size_per_head,
          size_t                              inter_size,
          size_t                              num_layer,
          size_t                              vocab_size,
          size_t                              rotary_embedding_dim,
          float                               layernorm_eps,
          int                                 start_id,
          int                                 end_id,
          int                                 prompt_learning_start_id,  // only needed by p/prompt-tuning
          bool                                use_gptj_residual,
          float                               beam_search_diversity_rate,
          size_t                              top_k,
          float                               top_p,
          unsigned long long                  random_seed,
          float                               temperature,
          float                               len_penalty,
          float                               repetition_penalty,
          bool                                is_free_buffer_after_forward,
          AttentionType                       attention_type           = AttentionType::UNFUSED_MHA);

    Llama(Llama<T> const& Llama);

    ~Llama();

    void forward(const LlamaWeight<T>*      gpt_weights);
};

}  // namespace fastertransformer
