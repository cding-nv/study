// flash_attn.cpp
#include <torch/extension.h>

void flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out);

void flash_attn_blockwise_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward, "Flash Attention Forward");
    m.def("flash_attn_blockwise_forward", &flash_attn_blockwise_forward, "Flash Attention blockwise Forward");
}

