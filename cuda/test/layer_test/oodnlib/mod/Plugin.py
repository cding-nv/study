url_AdjMatrixBatch_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FAdjMatrixBatch.h?Expires=4780469151&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=AqdfEZ0yXhHKa8sawbBT7Gg%2BYKc%3D"
# url_BatchingSequence_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FBatchingSequence.h?Expires=4780469217&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=M%2BZA6dl0GcuJ1nI3P%2Bo2jV4%2FBj0%3D"
url_BertEncoderInfer_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FBertEncoderInfer.h?Expires=4780469249&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=%2Ff%2FyY%2FYrHVDrMjAYW0ojlYycfI4%3D"
url_BertPoolerInfer_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FBertPoolerInfer.h?Expires=4780469279&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=scptsu%2BFCWM0Y1cOZUZIy2dD060%3D"
# url_MMSelfAttnInferL_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FMMSelfAttnInferL.h?Expires=4789073733&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=xCr0KOXPme%2F1%2FWPrqoPQqA%2BtVFw%3D"
# url_MMSelfAttnInferGL_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FMMSelfAttnInferGL.h?Expires=4789073677&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=QBdWZ2b6KI2KMXV1HsQ74cmQB4w%3D"
# url_PositionsAndTimeDiff_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FPositionsAndTimeDiff.h?Expires=4780469308&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=khg2ZlRl6ceH4F%2FNBg2m4gNMSf8%3D"
# url_RecoverSequenceInfer_h = "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FRecoverSequenceInfer.h?Expires=4780469334&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=oa5SRYNppbTMNy%2FsWZXPQOtPXEw%3D"
url_kernel_lib = [
    "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2FlibOODNLibKernel.so?Expires=4780469369&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=mYlP2ZcQNGKECH6UwxiK4LV3GIM%3D"
]
url_torch_lib = {
    "bert": "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2Fbert.so?Expires=4780469404&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=KTbYOo%2FhvJClij4LwvDH1F5ziU0%3D",
    # "mmsa": "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2Fmmsa.so?Expires=4789073618&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=E%2F8%2BD%2BXle49ZqAeXAMO2ecEbT1c%3D",
    "func": "http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/oodn-wheel%2Fplugin%2Ffunc.so?Expires=4780469451&OSSAccessKeyId=Oo2cqQNkidSaUBIN&Signature=BGZuhrhIy0txUAfCoR%2BBcDMvNpo%3D",
}


all_plugin_def = {
    "bert.encoder_infer": {
        "arg_types": [
            ["pre_layernorm", "bool"],
            ["num_attention_heads", "int"],
            ["intermediate_size", "int"],
            ["layer_norm_eps", "float"],
            ["fast_fp32", "bool"],
            ["num_hidden_layers", "int"],
        ],
        "required_dims": [
            {"name": "seq_len", "size": "inputs_0.1"},
            {"name": "hidden_size", "size": "inputs_0.2"},
            {"name": "input_numel", "size": "seq_len * hidden_size", "in_args": False},
            {
                "name": "probs_numel",
                "size": "num_attention_heads * seq_len * seq_len",
                "in_args": False,
            },
            {
                "name": "media_numel",
                "size": "seq_len * intermediate_size",
                "in_args": False,
            },
        ],
        "output_shapes": [
            "inputs_0",
            ["batch_size", "num_hidden_layers", "seq_len", "hidden_size"],
        ],
        "buffers": [
            {
                "name": "buffer",
                "size": "2 * input_numel + max(media_numel, input_numel + max(probs_numel, 3 * input_numel))",
                "dtype": "default",
            },
        ],
        "parameters": {"start": 2, "end": -1},
        "input_layouts": ["inputs_0", "any"],
        "output_layouts": ["inputs_0", "NCHW"],
        "cuda_header": url_BertEncoderInfer_h,
        "kernel_name": "BertEncoderInfer::forward",
        "tensorrt_type": "bert_encoder_infer",
    },
    "bert.pooler_infer": {
        "arg_types": [
            ["cls_count", "int"],
            ["fast_fp32", "bool"],
        ],
        "required_dims": [
            {"name": "seq_len", "size": "inputs_0.1"},
            {"name": "hidden_size", "size": "inputs_0.2"},
        ],
        "output_shapes": [
            ["batch_size", "cls_count", "hidden_size"],
        ],
        "buffers": [
            {"name": "buffer", "size": "cls_count * hidden_size", "dtype": "default"},
        ],
        "parameters": {"start": 1, "end": -1},
        "input_layouts": ["inputs_0"],
        "output_layouts": ["inputs_0"],
        "cuda_header": url_BertPoolerInfer_h,
        "kernel_name": "BertPoolerInfer::forward",
        "tensorrt_type": "bert_pooler_infer",
    },
    # "mmsa.MMSelfAttnInferL": {
    #     "arg_types": [
    #         ["use_multistream", "bool"],
    #         ["fast_fp32", "bool"],
    #     ],
    #     "required_dims": [
    #         {"name": "modal_cnt", "size": "inputs_0.0"},
    #         {"name": "seq_len", "size": "inputs_1.1"},
    #         {"name": "num_attention_heads", "size": "inputs_1.2"},
    #         {"name": "attention_head_size", "size": "inputs_1.3"},
    #         {
    #             "name": "hidden_size",
    #             "size": "num_attention_heads * attention_head_size",
    #             "in_args": False,
    #         },
    #     ],
    #     "output_shapes": [
    #         ["batch_size", "seq_len", "hidden_size"],
    #     ],
    #     "input_layouts": ["any", "NCHW", "NCHW", "NCHW", "any", "any"],
    #     "output_layouts": ["NHW"],
    #     "cuda_header": url_MMSelfAttnInferL_h,
    #     "kernel_name": "MMSelfAttnInferL::forward",
    #     "tensorrt_type": "mmsa_infer_l",
    # },
    # "mmsa.MMSelfAttnInferGL": {
    #     "arg_types": [
    #         ["use_multistream", "bool"],
    #         ["fast_fp32", "bool"],
    #     ],
    #     "required_dims": [
    #         {"name": "modal_cnt", "size": "inputs_0.0"},
    #         {"name": "seq_len", "size": "inputs_1.1"},
    #         {"name": "num_attention_heads", "size": "inputs_1.2"},
    #         {"name": "attention_head_size", "size": "inputs_1.3"},
    #         {"name": "max_num_global_indices_per_batch", "size": "inputs_5.2"},
    #         {"name": "global_selection_padding_mask_zeros_nRow", "size": "inputs_7.0"},
    #         {
    #             "name": "hidden_size",
    #             "size": "num_attention_heads * attention_head_size",
    #             "in_args": False,
    #         },
    #     ],
    #     "output_shapes": [
    #         ["batch_size", "seq_len", "hidden_size"],
    #     ],
    #     "input_layouts": ["any", "NCHW", "NCHW", "NCHW", "any", "NCHW", "NCHW", "any", "any"],
    #     "output_layouts": ["NHW"],
    #     "cuda_header": url_MMSelfAttnInferGL_h,
    #     "kernel_name": "MMSelfAttnInferGL::forward",
    #     "tensorrt_type": "mmsa_infer_gl",
    # },
    "func.AdjMatrixBatchSimpleGenerate": {
        "arg_types": [
            ["alpha", "float"],
        ],
        "required_dims": [
            {"name": "seq_len", "size": "inputs_0.1"},
        ],
        "output_shapes": [["batch_size", "seq_len", "seq_len"]],
        "input_layouts": ["inputs_0"],
        "output_layouts": ["NHW"],
        "cuda_header": url_AdjMatrixBatch_h,
        "kernel_name": "launchAdjMatrixBatch",
        "tensorrt_type": "func_adj_matrix_batch_simple_generate",
    },
    # "func.BatchingSequenceOfSequenceDataReduceInvalidInputA": {
    #     "cuda_header": url_BatchingSequence_h,
    #     "kernel_name": "launchMaskLength",
    #     "tensorrt_type": "func_batching_sequence_of_sequence_data_reduce_invalid_input_a"
    # },
    # "func.BatchingSequenceOfSequenceDataReduceInvalidInputB": {
    #     "cuda_header": url_BatchingSequence_h,
    #     "kernel_name": "launchBatchingSequence",
    #     "tensorrt_type": "func_batching_sequence_of_sequence_data_reduce_invalid_input_b"
    # },
    # "func.PositionsAndTimeDiff": {
    #     "cuda_header": url_PositionsAndTimeDiff_h,
    #     "kernel_name": "PositionsAndTimeDiff",
    #     "tensorrt_type": "func_positions_and_time_diff"
    # },
    # "func.RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer": {
    #     "cuda_header": url_RecoverSequenceInfer_h,
    #     "kernel_name": "RecoverSequenceInfer::forward1D",
    #     "tensorrt_type": "func_recover_sequence_of_sequence_data_reduce_invalid_input_1d_infer"
    # },
    # "func.RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer": {
    #     "cuda_header": url_RecoverSequenceInfer_h,
    #     "kernel_name": "RecoverSequenceInfer::forward2D",
    #     "tensorrt_type": "func_recover_sequence_of_sequence_data_reduce_invalid_input_2d_infer"
    # },
}


loaded_ops = set()


def register_op(name):
    loaded_ops.add(name)


def get_loaded_ops():
    if len(loaded_ops) == 0:
        return None
    return loaded_ops


def get_plugin_info():
    if len(loaded_ops) == 0:
        return None

    torch_libs = set()
    plugin_def = {}
    for name in loaded_ops:
        torch_libs.add(url_torch_lib[name.split(".")[0]])
        plugin_def[name] = all_plugin_def[name]
    torch_libs = list(torch_libs)

    plugin_info = {
        "plugin_device": "gpu",
        "torch_libs": torch_libs,
        "cuda_libs": url_kernel_lib,
        "plugin_def": plugin_def,
    }
    return plugin_info
