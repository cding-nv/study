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

#include "src/xfastertransformer/models/llama/Llama.h"

#include <algorithm>

namespace fastertransformer {

template<typename T>
void Llama<T>::initialize()
{
    printf("#### init Llama.\n");

}

template<typename T>
void Llama<T>::allocateBuffer(
    size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_cache_seq_len, size_t max_input_len)
{
    
}

template<typename T>
void Llama<T>::freeBuffer() { 
}

template<typename T>
Llama<T>::Llama(size_t                              head_num,
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
                AttentionType                       attention_type):
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    layernorm_eps_(layernorm_eps),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    //prompt_learning_type_(prompt_learning_type),
    use_gptj_residual_(use_gptj_residual),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num / 1),
    attention_type_(attention_type)
{
    initialize();
}

template<typename T>
Llama<T>::~Llama()
{
    freeBuffer();
}

template<typename T>
void Llama<T>::forward(const LlamaWeight<T>*                        gpt_weights)
{
    

    printf("#### Llama forward.\n");
   
}


template class Llama<float>;
//template class Llama<half>;

}  // namespace fastertransformer
