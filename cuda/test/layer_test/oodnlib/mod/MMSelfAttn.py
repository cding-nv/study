import torch
import oodnlib.ext.mmsa
from .Plugin import register_op
from torch import nn
from typing import List


def _align_numel(numel: int, elsize: int) -> int:
    numel_align: int = ((numel * elsize + 0xF) & 0x7FFFFFFFFFFFFFF0) // elsize
    return numel_align


@torch.jit.script
def get_inferL_buff(
    modal_index: List[List[int]],
    key_layer: torch.Tensor,  # [B, S, N, A]
    use_multistream: bool,
) -> torch.Tensor:
    bn: int = key_layer.size(0) * key_layer.size(2)
    bna: int = bn * key_layer.size(3)
    elsize: int = key_layer.element_size()
    seq_lens: List[int] = [idx[1] - idx[0] for idx in modal_index]
    qkvL_numels: List[int] = [bna * s for s in seq_lens]
    attnL_numels: List[int] = [_align_numel(bn * s * s, elsize) for s in seq_lens]
    buff_sizes: List[int] = [X * 2 + L for X, L in zip(qkvL_numels, attnL_numels)]
    inferL_buff_size: int = 0
    if use_multistream:
        for size in buff_sizes:
            inferL_buff_size += size
    else:
        for size in buff_sizes:
            inferL_buff_size = max(inferL_buff_size, size)
    inferL_buff = torch.empty(
        inferL_buff_size, dtype=key_layer.dtype, device=key_layer.device
    )
    return inferL_buff


@torch.jit.script
def get_inferGL_buff(
    modal_index: List[List[int]],
    key_layer: torch.Tensor,  # [B, S, N, A]
    global_k: torch.Tensor,  # [B, N, S, A]
    use_multistream: bool,
) -> torch.Tensor:
    bn: int = key_layer.size(0) * key_layer.size(2)
    gs: int = global_k.size(2)
    bna: int = bn * key_layer.size(3)
    bns: int = bn * gs
    elsize: int = key_layer.element_size()
    seq_lens: List[int] = [idx[1] - idx[0] for idx in modal_index]
    qkvL_numels: List[int] = [bna * s for s in seq_lens]
    attnL_numels: List[int] = [_align_numel(bn * s * s, elsize) for s in seq_lens]
    attnG_numels: List[int] = [_align_numel(bns * s, elsize) for s in seq_lens]
    attnGL_numels: List[int] = [
        _align_numel(bn * s * (s + gs), elsize) for s in seq_lens
    ]
    buff_sizes: List[int] = [
        L + max(X, G) + max(X, GL)
        for X, L, G, GL in zip(qkvL_numels, attnL_numels, attnG_numels, attnGL_numels)
    ]
    inferGL_buff_size: int = 0
    if use_multistream:
        for size in buff_sizes:
            inferGL_buff_size += size
    else:
        for size in buff_sizes:
            inferGL_buff_size = max(inferGL_buff_size, size)
    inferGL_buff = torch.empty(
        inferGL_buff_size, dtype=key_layer.dtype, device=key_layer.device
    )
    return inferGL_buff


class MMSelfAttn(nn.Module):
    def __init__(
        self, use_multistream, fast_fp32, modal_cnt, attention_probs_dropout_prob
    ):
        super(MMSelfAttn, self).__init__()
        stream_cnt = modal_cnt if use_multistream else 1
        torch.ops.mmsa.MMSelfAttn_init(stream_cnt, fast_fp32)
        self.use_multistream = use_multistream
        self.fast_fp32 = fast_fp32
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    def forward(
        self,
        modal_index: List[List[int]],
        query_layer: torch.Tensor,  # [B, S, N, A], contiguous
        key_layer: torch.Tensor,  # [B, S, N, A], contiguous
        value_layer: torch.Tensor,  # [B, S, N, A], contiguous
        local_attention_mask: torch.Tensor,
        global_attention_mask_is_not_None: bool,
        global_k: torch.Tensor,  # [B, N, S, A]
        global_v: torch.Tensor,  # [B, N, S, A]
        global_selection_padding_mask_zeros: torch.Tensor,
    ) -> torch.Tensor:
        # modify inputs
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        if global_attention_mask_is_not_None:
            if self.training:
                context_layer = torch.ops.mmsa.MMSelfAttnTrainGL(
                    torch.tensor(modal_index),
                    query_layer,
                    key_layer,
                    value_layer,
                    local_attention_mask,
                    global_k,
                    global_v,
                    global_selection_padding_mask_zeros,
                    self.attention_probs_dropout_prob,
                    self.use_multistream,
                )
            else:
                # register_op("mmsa.MMSelfAttnInferGL")
                inferGL_buff = get_inferGL_buff(
                    modal_index, key_layer, global_k, self.use_multistream
                )
                context_layer = torch.ops.mmsa.MMSelfAttnInferGL(
                    torch.tensor(modal_index),
                    query_layer,
                    key_layer,
                    value_layer,
                    local_attention_mask,
                    global_k,
                    global_v,
                    global_selection_padding_mask_zeros,
                    inferGL_buff,
                    self.use_multistream,
                    self.fast_fp32,
                )
        else:
            if self.training:
                context_layer = torch.ops.mmsa.MMSelfAttnTrainL(
                    torch.tensor(modal_index),
                    query_layer,
                    key_layer,
                    value_layer,
                    local_attention_mask,
                    self.attention_probs_dropout_prob,
                    self.use_multistream,
                )
            else:
                # register_op("mmsa.MMSelfAttnInferL")
                inferL_buff = get_inferL_buff(
                    modal_index, key_layer, self.use_multistream
                )
                context_layer = torch.ops.mmsa.MMSelfAttnInferL(
                    torch.tensor(modal_index),
                    query_layer,
                    key_layer,
                    value_layer,
                    local_attention_mask,
                    inferL_buff,
                    self.use_multistream,
                    self.fast_fp32,
                )
        return context_layer
