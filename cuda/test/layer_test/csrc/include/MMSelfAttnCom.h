#pragma once


class MMSelfAttnCom {
private:
    const int64_t* _modal_index = nullptr;
    int64_t _modal_cnt = 0ULL;
    int64_t _batch_size = 0ULL;
    int64_t _hidden_size = 0ULL;
    int64_t _num_attention_heads = 0ULL;
    int64_t _max_num_global_indices_per_batch = 0ULL;

private:
    template <typename T>
    int64_t
    align_numel(int64_t numel) const {
        int64_t numel_align = ((numel * sizeof(T) + 0xF) & 0xFFFFFFFFFFFFFFF0) / sizeof(T);
        return numel_align;
    }

public:
    MMSelfAttnCom(const int64_t* modal_index,
                  int64_t modal_cnt,
                  int64_t batch_size,
                  int64_t num_attention_heads,
                  int64_t attention_head_size,
                  int64_t max_num_global_indices_per_batch = 0ULL):
        _modal_index(modal_index),
        _modal_cnt(modal_cnt),
        _batch_size(batch_size),
        _hidden_size(num_attention_heads * attention_head_size),
        _num_attention_heads(num_attention_heads),
        _max_num_global_indices_per_batch(max_num_global_indices_per_batch)
    {}

    int64_t
    get_modal_cnt() const {
        return _modal_cnt;
    }

    int64_t
    get_index_begin(int64_t idx) const {
        int64_t index_begin = _modal_index[2 * idx];
        return index_begin;
    }

    int64_t
    get_current_seq_len(int64_t idx) const {
        int64_t current_seq_len = _modal_index[2 * idx + 1] - _modal_index[2 * idx];
        return current_seq_len;
    }

    int64_t
    get_current_local_qkv_numel(int64_t idx) const {
        int64_t current_seq_len = get_current_seq_len(idx);
        int64_t current_local_qkv_numel = _batch_size * current_seq_len * _hidden_size;
        return current_local_qkv_numel;
    }

    template <typename T>
    int64_t
    get_current_local_attn_numel_align(int64_t idx) const {
        int64_t current_seq_len = get_current_seq_len(idx);
        int64_t current_local_attn_numel = _batch_size * _num_attention_heads * current_seq_len * current_seq_len;
        int64_t current_local_attn_numel_align = align_numel<T>(current_local_attn_numel);
        return current_local_attn_numel_align;
    }

    template <typename T>
    int64_t
    get_current_global_attn_numel_align(int64_t idx) const {
        int64_t current_seq_len = get_current_seq_len(idx);
        int64_t current_global_attn_numel = _batch_size * _num_attention_heads * current_seq_len * _max_num_global_indices_per_batch;
        int64_t current_global_attn_numel_align = align_numel<T>(current_global_attn_numel);
        return current_global_attn_numel_align;
    }

    int64_t
    get_current_attn_numel(int64_t idx) const {
        int64_t current_seq_len = get_current_seq_len(idx);
        int64_t current_attn_numel = _batch_size * _num_attention_heads * current_seq_len * (current_seq_len + _max_num_global_indices_per_batch);
        return current_attn_numel;
    }

    template <typename T>
    int64_t
    get_current_attn_numel_align(int64_t idx) const {
        int64_t current_attn_numel_align = align_numel<T>(get_current_attn_numel(idx));
        return current_attn_numel_align;
    }

    int64_t
    get_local_qkv_numel() const {
        int64_t local_qkv_numel = 0LL;
        for (int64_t i = 0LL; i < _modal_cnt; ++i) {
            local_qkv_numel += get_current_local_qkv_numel(i);
        }
        return local_qkv_numel;
    }

    int64_t
    get_local_qkv_numel_max() const {
        int64_t local_qkv_numel_max = 0LL;
        for (int64_t i = 0LL; i < _modal_cnt; ++i) {
            local_qkv_numel_max = max(local_qkv_numel_max, get_current_local_qkv_numel(i));
        }
        return local_qkv_numel_max;
    }

    int64_t
    get_global_qkv_numel() const {
        int64_t global_qkv_numel = _batch_size * _max_num_global_indices_per_batch * _hidden_size;
        return global_qkv_numel;
    }

    template <typename T>
    int64_t
    get_local_attn_numel_align() const {
        int64_t local_attn_numel_align = 0LL;
        for (int64_t i = 0LL; i < _modal_cnt; ++i) {
            local_attn_numel_align += get_current_local_attn_numel_align<T>(i);
        }
        return local_attn_numel_align;
    }

    template <typename T>
    int64_t
    get_global_attn_numel_align() const {
        int64_t global_attn_numel_align = 0LL;
        for (int64_t i = 0LL; i < _modal_cnt; ++i) {
            global_attn_numel_align += get_current_global_attn_numel_align<T>(i);
        }
        return global_attn_numel_align;
    }

    template <typename T>
    int64_t
    get_attn_numel_align() const {
        int64_t attn_numel_align = 0LL;
        for (int64_t i = 0LL; i < _modal_cnt; ++i) {
            attn_numel_align += get_current_attn_numel_align<T>(i);
        }
        return attn_numel_align;
    }
};

