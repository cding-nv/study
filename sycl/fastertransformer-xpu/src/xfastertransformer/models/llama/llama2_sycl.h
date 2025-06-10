#include "common/common.h"
#include "common/sampler.h"

namespace fastertransformer {
void llama_sycl_init(Transformer* transformer, Sampler* sampler, char* model_path);
void llama_sycl_forward(Transformer* transformer, Sampler* sampler, int* output_tokens, int output_len, int* prompt_tokens, int input_len);
}
