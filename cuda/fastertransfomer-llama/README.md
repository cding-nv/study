# FasterTransformer - llama2

This repository provides a script and recipe to run the highly optimized transformer-based encoder and decoder component.

## llama cpp example
```
$  sudo docker run --gpus all -ti --net=host --shm-size 5g --ulimit memlock=-1 --rm nvcr.io/nvidia/pytorch:23.02-py3 bash
$  pip install transformers==4.29.2  sentencepiece bfloat16
$  mkdir build && cd build
$  cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release ..
$  make -j

```

## Modify llama_config.ini and Run
```
$ ./bin/llama_example
```

## More details

1. Convert model for single gpu
```
$ cd example/cpp/llama
$ python huggingface_llama_convert.py -saved_dir=./output -in_file=/home/fengding/Llama-2-7b-hf/ -infer_gpu_num=1 -weight_data_type=fp16 -model_name=llama_7b
```
The ouput is like the below
```
config.ini                                              model.layers.24.attention.dense.weight.0.bin
model.final_layernorm.weight.bin                        model.layers.24.attention.query_key_value.weight.0.bin
model.layers.0.attention.dense.weight.0.bin             model.layers.24.input_layernorm.weight.bin
model.layers.0.attention.query_key_value.weight.0.bin   model.layers.24.mlp.down_proj.weight.0.bin
model.layers.0.input_layernorm.weight.bin               model.layers.24.mlp.gate_proj.weight.0.bin
model.layers.0.mlp.down_proj.weight.0.bin               model.layers.24.mlp.up_proj.weight.0.bin
model.layers.0.mlp.gate_proj.weight.0.bin               model.layers.24.post_attention_layernorm.weight.bin
model.layers.0.mlp.up_proj.weight.0.bin                 model.layers.25.attention.dense.weight.0.bin
model.layers.0.post_attention_layernorm.weight.bin      model.layers.25.attention.query_key_value.weight.0.bin
model.layers.10.attention.dense.weight.0.bin            model.layers.25.input_layernorm.weight.bin
model.layers.10.attention.query_key_value.weight.0.bin  model.layers.25.mlp.down_proj.weight.0.bin
model.layers.10.input_layernorm.weight.bin              model.layers.25.mlp.gate_proj.weight.0.bin
model.layers.10.mlp.down_proj.weight.0.bin              model.layers.25.mlp.up_proj.weight.0.bin
model.layers.10.mlp.gate_proj.weight.0.bin              model.layers.25.post_attention_layernorm.weight.bin
model.layers.10.mlp.up_proj.weight.0.bin                model.layers.26.attention.dense.weight.0.bin
model.layers.10.post_attention_layernorm.weight.bin     model.layers.26.attention.query_key_value.weight.0.bin
model.layers.11.attention.dense.weight.0.bin            model.layers.26.input_layernorm.weight.bin
model.layers.11.attention.query_key_value.weight.0.bin  model.layers.26.mlp.down_proj.weight.0.bin
model.layers.11.input_layernorm.weight.bin              model.layers.26.mlp.gate_proj.weight.0.bin
model.layers.11.mlp.down_proj.weight.0.bin              model.layers.26.mlp.up_proj.weight.0.bin
model.layers.11.mlp.gate_proj.weight.0.bin              model.layers.26.post_attention_layernorm.weight.bin
model.layers.11.mlp.up_proj.weight.0.bin                model.layers.27.attention.dense.weight.0.bin
model.layers.11.post_attention_layernorm.weight.bin     model.layers.27.attention.query_key_value.weight.0.bin
model.layers.12.attention.dense.weight.0.bin            model.layers.27.input_layernorm.weight.bin
model.layers.12.attention.query_key_value.weight.0.bin  model.layers.27.mlp.down_proj.weight.0.bin
model.layers.12.input_layernorm.weight.bin              model.layers.27.mlp.gate_proj.weight.0.bin
model.layers.12.mlp.down_proj.weight.0.bin              model.layers.27.mlp.up_proj.weight.0.bin
model.layers.12.mlp.gate_proj.weight.0.bin              model.layers.27.post_attention_layernorm.weight.bin
model.layers.12.mlp.up_proj.weight.0.bin                model.layers.28.attention.dense.weight.0.bin
model.layers.12.post_attention_layernorm.weight.bin     model.layers.28.attention.query_key_value.weight.0.bin
model.layers.13.attention.dense.weight.0.bin            model.layers.28.input_layernorm.weight.bin
model.layers.13.attention.query_key_value.weight.0.bin  model.layers.28.mlp.down_proj.weight.0.bin
model.layers.13.input_layernorm.weight.bin              model.layers.28.mlp.gate_proj.weight.0.bin
model.layers.13.mlp.down_proj.weight.0.bin              model.layers.28.mlp.up_proj.weight.0.bin
model.layers.13.mlp.gate_proj.weight.0.bin              model.layers.28.post_attention_layernorm.weight.bin
model.layers.13.mlp.up_proj.weight.0.bin                model.layers.29.attention.dense.weight.0.bin
model.layers.13.post_attention_layernorm.weight.bin     model.layers.29.attention.query_key_value.weight.0.bin
model.layers.14.attention.dense.weight.0.bin            model.layers.29.input_layernorm.weight.bin
model.layers.14.attention.query_key_value.weight.0.bin  model.layers.29.mlp.down_proj.weight.0.bin
model.layers.14.input_layernorm.weight.bin              model.layers.29.mlp.gate_proj.weight.0.bin
model.layers.14.mlp.down_proj.weight.0.bin              model.layers.29.mlp.up_proj.weight.0.bin
model.layers.14.mlp.gate_proj.weight.0.bin              model.layers.29.post_attention_layernorm.weight.bin
model.layers.14.mlp.up_proj.weight.0.bin                model.layers.2.attention.dense.weight.0.bin
model.layers.14.post_attention_layernorm.weight.bin     model.layers.2.attention.query_key_value.weight.0.bin
model.layers.15.attention.dense.weight.0.bin            model.layers.2.input_layernorm.weight.bin
model.layers.15.attention.query_key_value.weight.0.bin  model.layers.2.mlp.down_proj.weight.0.bin
model.layers.15.input_layernorm.weight.bin              model.layers.2.mlp.gate_proj.weight.0.bin
model.layers.15.mlp.down_proj.weight.0.bin              model.layers.2.mlp.up_proj.weight.0.bin
model.layers.15.mlp.gate_proj.weight.0.bin              model.layers.2.post_attention_layernorm.weight.bin
model.layers.15.mlp.up_proj.weight.0.bin                model.layers.30.attention.dense.weight.0.bin
model.layers.15.post_attention_layernorm.weight.bin     model.layers.30.attention.query_key_value.weight.0.bin
model.layers.16.attention.dense.weight.0.bin            model.layers.30.input_layernorm.weight.bin
model.layers.16.attention.query_key_value.weight.0.bin  model.layers.30.mlp.down_proj.weight.0.bin
model.layers.16.input_layernorm.weight.bin              model.layers.30.mlp.gate_proj.weight.0.bin
model.layers.16.mlp.down_proj.weight.0.bin              model.layers.30.mlp.up_proj.weight.0.bin
model.layers.16.mlp.gate_proj.weight.0.bin              model.layers.30.post_attention_layernorm.weight.bin
model.layers.16.mlp.up_proj.weight.0.bin                model.layers.31.attention.dense.weight.0.bin
model.layers.16.post_attention_layernorm.weight.bin     model.layers.31.attention.query_key_value.weight.0.bin
model.layers.17.attention.dense.weight.0.bin            model.layers.31.input_layernorm.weight.bin
model.layers.17.attention.query_key_value.weight.0.bin  model.layers.31.mlp.down_proj.weight.0.bin
model.layers.17.input_layernorm.weight.bin              model.layers.31.mlp.gate_proj.weight.0.bin
model.layers.17.mlp.down_proj.weight.0.bin              model.layers.31.mlp.up_proj.weight.0.bin
model.layers.17.mlp.gate_proj.weight.0.bin              model.layers.31.post_attention_layernorm.weight.bin
model.layers.17.mlp.up_proj.weight.0.bin                model.layers.3.attention.dense.weight.0.bin
model.layers.17.post_attention_layernorm.weight.bin     model.layers.3.attention.query_key_value.weight.0.bin
model.layers.18.attention.dense.weight.0.bin            model.layers.3.input_layernorm.weight.bin
model.layers.18.attention.query_key_value.weight.0.bin  model.layers.3.mlp.down_proj.weight.0.bin
model.layers.18.input_layernorm.weight.bin              model.layers.3.mlp.gate_proj.weight.0.bin
model.layers.18.mlp.down_proj.weight.0.bin              model.layers.3.mlp.up_proj.weight.0.bin
model.layers.18.mlp.gate_proj.weight.0.bin              model.layers.3.post_attention_layernorm.weight.bin
model.layers.18.mlp.up_proj.weight.0.bin                model.layers.4.attention.dense.weight.0.bin
model.layers.18.post_attention_layernorm.weight.bin     model.layers.4.attention.query_key_value.weight.0.bin
model.layers.19.attention.dense.weight.0.bin            model.layers.4.input_layernorm.weight.bin
model.layers.19.attention.query_key_value.weight.0.bin  model.layers.4.mlp.down_proj.weight.0.bin
model.layers.19.input_layernorm.weight.bin              model.layers.4.mlp.gate_proj.weight.0.bin
model.layers.19.mlp.down_proj.weight.0.bin              model.layers.4.mlp.up_proj.weight.0.bin
model.layers.19.mlp.gate_proj.weight.0.bin              model.layers.4.post_attention_layernorm.weight.bin
model.layers.19.mlp.up_proj.weight.0.bin                model.layers.5.attention.dense.weight.0.bin
model.layers.19.post_attention_layernorm.weight.bin     model.layers.5.attention.query_key_value.weight.0.bin
model.layers.1.attention.dense.weight.0.bin             model.layers.5.input_layernorm.weight.bin
model.layers.1.attention.query_key_value.weight.0.bin   model.layers.5.mlp.down_proj.weight.0.bin
model.layers.1.input_layernorm.weight.bin               model.layers.5.mlp.gate_proj.weight.0.bin
model.layers.1.mlp.down_proj.weight.0.bin               model.layers.5.mlp.up_proj.weight.0.bin
model.layers.1.mlp.gate_proj.weight.0.bin               model.layers.5.post_attention_layernorm.weight.bin
model.layers.1.mlp.up_proj.weight.0.bin                 model.layers.6.attention.dense.weight.0.bin
model.layers.1.post_attention_layernorm.weight.bin      model.layers.6.attention.query_key_value.weight.0.bin
model.layers.20.attention.dense.weight.0.bin            model.layers.6.input_layernorm.weight.bin
model.layers.20.attention.query_key_value.weight.0.bin  model.layers.6.mlp.down_proj.weight.0.bin
model.layers.20.input_layernorm.weight.bin              model.layers.6.mlp.gate_proj.weight.0.bin
model.layers.20.mlp.down_proj.weight.0.bin              model.layers.6.mlp.up_proj.weight.0.bin
model.layers.20.mlp.gate_proj.weight.0.bin              model.layers.6.post_attention_layernorm.weight.bin
model.layers.20.mlp.up_proj.weight.0.bin                model.layers.7.attention.dense.weight.0.bin
model.layers.20.post_attention_layernorm.weight.bin     model.layers.7.attention.query_key_value.weight.0.bin
model.layers.21.attention.dense.weight.0.bin            model.layers.7.input_layernorm.weight.bin
model.layers.21.attention.query_key_value.weight.0.bin  model.layers.7.mlp.down_proj.weight.0.bin
model.layers.21.input_layernorm.weight.bin              model.layers.7.mlp.gate_proj.weight.0.bin
model.layers.21.mlp.down_proj.weight.0.bin              model.layers.7.mlp.up_proj.weight.0.bin
model.layers.21.mlp.gate_proj.weight.0.bin              model.layers.7.post_attention_layernorm.weight.bin
model.layers.21.mlp.up_proj.weight.0.bin                model.layers.8.attention.dense.weight.0.bin
model.layers.21.post_attention_layernorm.weight.bin     model.layers.8.attention.query_key_value.weight.0.bin
model.layers.22.attention.dense.weight.0.bin            model.layers.8.input_layernorm.weight.bin
model.layers.22.attention.query_key_value.weight.0.bin  model.layers.8.mlp.down_proj.weight.0.bin
model.layers.22.input_layernorm.weight.bin              model.layers.8.mlp.gate_proj.weight.0.bin
model.layers.22.mlp.down_proj.weight.0.bin              model.layers.8.mlp.up_proj.weight.0.bin
model.layers.22.mlp.gate_proj.weight.0.bin              model.layers.8.post_attention_layernorm.weight.bin
model.layers.22.mlp.up_proj.weight.0.bin                model.layers.9.attention.dense.weight.0.bin
model.layers.22.post_attention_layernorm.weight.bin     model.layers.9.attention.query_key_value.weight.0.bin
model.layers.23.attention.dense.weight.0.bin            model.layers.9.input_layernorm.weight.bin
model.layers.23.attention.query_key_value.weight.0.bin  model.layers.9.mlp.down_proj.weight.0.bin
model.layers.23.input_layernorm.weight.bin              model.layers.9.mlp.gate_proj.weight.0.bin
model.layers.23.mlp.down_proj.weight.0.bin              model.layers.9.mlp.up_proj.weight.0.bin
model.layers.23.mlp.gate_proj.weight.0.bin              model.layers.9.post_attention_layernorm.weight.bin
model.layers.23.mlp.up_proj.weight.0.bin                model.lm_head.weight.bin
model.layers.23.post_attention_layernorm.weight.bin     model.wte.weight.bin
```

2. Reference    
cpp: https://github.com/NVIDIA/FasterTransformer/pull/575    
pytorch: https://github.com/NVIDIA/FasterTransformer/pull/611    

3. Code reading

gpt
```
git checkout dd4c071755b1b5206c6987add6145eac03f6814d
gpt_sample.cc
    decoding_sample
       mpi, nccl
       *decoding = new DecodingGpt<type>
       decoding->forward_context
       decoding->forward
           gpt.h DecodingGpt

https://github.com/NVIDIA/FasterTransformer/blob/dev/v5.0_beta/docs/gpt_guide.md

./sample/pytorch/gpt_sample.py
./sample/tensorflow/gpt_sample.py

pytorch gpt_sample.py
   gpt.py   
        class GPT(nn.Module):
            dist.init_process_group(backend='mpi')
       self.weights.load    load 对应 part 的 weights
       self.model = torch.classes.FasterTransformer.GPT
       torch.classes.load_library(os.path.abspath(lib_path))
           libpyt_fastertransformer.so
               ft_op.cc
                torch::jit::class_<torch_ext::FasterTransformerGPT>("FasterTransformer", "GPT")
                   .def("forward",&torch_ext::FasterTransformerGPT::forward)  ->  gpt->forward
                   torch::RegisterOperators("fastertransformer::build_mask_remove_padding", &torch_ext::build_mask_remove_padding);
                   torch::RegisterOperators("fastertransformer::rebuild_padding", &torch_ext::rebuild_padding);
                   torch::RegisterOperators("fastertransformer::gather_tree", &torch_ext::gather_tree);
                   torch::RegisterOperators("fastertransformer::weight_quantize", &torch_ext::weight_quantize);
                   "FasterTransformer", "Decoder"  .def("forward") 
                   "FasterTransformer", "Encoder"  .def("forward")
                      gpt.cc / th_op/gpt.h
                                 FasterTransformerGPT::forward
                       gpt.h    class GPT
                         DecodingGpt    nccl_recv()  all2all_gather
                             open_decoder.h -> OpenDecoder ->  masked_multi_head_attention() all2all_reduce_sum
                                                               cross_multi_head_attention
                                                               ffn()   all2all_reduce_sum
                             forward_context()   nccl_recv  all2all_gather decoder_->forward_context nccl_send
                             forward()  nccl_broadcast  nccl_recv  all2all_gather  nccl_send
                         forward()  -> *decoding = new DecodingGpt  decoding->forward_context   decoding->forward
                           encoder.cc
                           decoder.cc
                           decoding.cc
                           utils.cu
                           weight_quantize_op.cc 
           output_ids, = self.model.forward(start_ids, start_lengths, attn_mask, self.output_len)
 
encoder_sample.py
       class CustomEncoder          对比 HuggingFaceEncoder
               encoder.py  -> torch.classes.FasterTransformer.Encoder
              libpyt_fastertransformer.so
                  ft_op.cc
                  torch::jit::class_<torch_ext::FasterTransformerEncoder>("FasterTransformer", "Encoder")
                     .def("forward", &torch_ext::FasterTransformerEncoder::forward)
                     th_op/encoder.h encoder.cc -> FasterTransformerEncoder -> ftencoder->forward -> th_op/encoder.h  FTEncoder->BertEncoderTransformer class 
         forward()
 
decoder_sample.py
    class CustomDecoder   对比 ONMTDecoder
        decoder.py  ->  torch.classes.FasterTransformer.Decoder
        libpyt_fastertransformer.so
            ft_op.cc
                torch::jit::class_<torch_ext::FasterTransformerDecoder>("FasterTransformer", "Decoder")
                .def("forward", &torch_ext::FasterTransformerDecoder::forward)
                th_op/decoder.h decoder.cc -> FasterTransformerDecoder -> ftdecoder->forward -> th_op/decoder.h
```

Llama 
```
	
	Llama.forward()
	   
	1. Gpt_kernel.cu 
	         invokeTileGptInputs -> tileGptPromptInputs()
	    start_id_embedding_position_lookups_kernel
	    invokeBuildDecoderAttentionMask
	
	2. gpt_context_decoder_ -> LlamaContextDecoder
	     llamaContextDecoder.cc
	            
	            TensorParallelSiluFfnLayer
	            invokeGetPaddingOffsetAndCuSeqLens -> bert_preprocess_kernels.cu
	                            invokeRemovePadding -> bert_preprocess_kernels.cu
	                            invokeGeneralT5LayerNorm ->  layernorm_kernels.cu
	            TensorParallelGptContextAttentionLayer ->GptContextAttentionLayer<T>::forward()
	                  gptContextAttentionLayer.cc
	                       cublas -> SpGemm
	                       invokeAddFusedQKVBiasTranspose -> unfused_attention_kernels.cu
	                                                     invokeTranspose4dBatchMajor    -> unfused_attention_kernels.cu
	
						attention_type 
						enum class AttentionType {
						    UNFUSED_MHA,
						    UNFUSED_PADDED_MHA,
						    FUSED_MHA,
						    FUSED_PADDED_MHA
						};
						Cublas -> stridedBatchedGemm
						invokeTransposeQKV -> unfused_attention_kernels.cu
						invokeTransposeAttentionOutRemovePadding -> unfused_attention_kernels.cu
				
				TensorParallelSiluFfnLayer -> forward()
				    TensorPparallelSiluFpnLayer.cc
				          SiluFfnLayer  (remove all reduce)  -> ffnLayer.cc 
				               cublas -> Gemm
				               invokeAddBiasGeluV2 -> activation_kernels.cu
				               invokeGenericActivation -> activation_kernels.cu
				
	     decoding_kernels.cu
	
	3. invokeDecodingInitialize ->decoding_kernels.cu decodingInitialize
	
	4. invokeMaskPaddingTokens -> Gpt_kernel.cu 
	
	5. invokeEmbeddingLookupPosEncodingPadCount -> decoding_kernels.cu
	
	
	6. gpt_decoder_->forward  -> LlamaDecoder
                   LlamaContextDecoder
                               TensorParallelGptContextAttentionLayer
                                      GptContextAttentionLayer
                                                  qkv_gemm,   cublas_wrapper_->Gemm
                           Q*K batch gemm,  cublas_wrapper_->stridedBatchedGemm
                           maskedSoftmax
                           QK*V batch gemm
                           proj gemm,  attention wo
                               TensorParallelSiluFfnLayer

	      llamaDecoder.cc
	            TensorParallelDecoderSelfAttentionLayer
	                                  DecoderSelfAttentionLayer<T>::forward()  (remove allreduce)
	                        cublas -> ge
	                        fusedQKV_masked_attention_dispatch -> decoderSelfAtentionLayer.cc
	                                masked_multihead_attention() -> decoder_masked_multihead_attention.cu
	                                     Decoder_masked_multihead_attention_128.cu
	                                     Decoder_masked_multihead_attention_80.cu
	                                     Decoder_masked_multihead_attention_96.cu
	                                     …….. 
	                                                                        decoder_masked_multihead_attention_template.hpp
	                            TensorParallelSiluFfnLayer  重复上面的
	
	
	
	Cublas->Gemm
	
	
	1. invokeGeneralT5LayerNorm -> layernorm_kernels.cu
	
	
	2. dynamic_decode_layer_->forward -> DynamicDecodeLayer
	         OnlineBeamSearchLayer -> OnlineBeamSearchLayer.cu
	         BeamSearchLayer -> BaseBeamSearchLayer.cu
	                     TopKSamplingLayer -> TopKSamplingLayer.cu
	                     TopPSamplingLayer -> TopPSamplingLayer.cu
	           
	         invokeBanBadWords -> ban_bad_words.cu
                     invokeStopWordsCriterion / invokeLengthCriterion -> stop_criteria_kernels.cu
```
