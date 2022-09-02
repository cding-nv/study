#pragma once


class BertCom {
public:
    static bool
    is_reusable(int64_t hidden_size,
                int64_t intermediate_size) {
        return intermediate_size >= hidden_size * 3;
    }

    static int64_t
    get_input_numel(int64_t batch_size,
                    int64_t seq_len,
                    int64_t hidden_size) {
        int64_t input_numel = batch_size * seq_len * hidden_size;
        return input_numel;
    }

    static int64_t
    get_probs_numel(int64_t batch_size,
                    int64_t seq_len,
                    int64_t num_attention_heads) {
        int64_t probs_numel = batch_size * num_attention_heads * seq_len * seq_len;
        return probs_numel;
    }

    static int64_t
    get_media_numel(int64_t batch_size,
                    int64_t seq_len,
                    int64_t intermediate_size) {
        int64_t media_numel = batch_size * seq_len * intermediate_size;
        return media_numel;
    }

    static int64_t
    get_output_numel(int64_t batch_size,
                     int64_t hidden_size,
                     int64_t cls_count) {
        int64_t output_numel = batch_size * cls_count * hidden_size;
        return output_numel;
    }
};

