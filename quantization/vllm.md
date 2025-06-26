
# Notes

### Linear method quant apply 什么时候调用？
```
vllm/engine/llm_engine.py(276)__init__()
-> self._initialize_kv_caches()
   Llama.py   LlamaAttention forward()  
    self.qkv_proj(hidden_states)
             vllm/model_executor/layers/linear.py(391)forward()
                 -> output_parallel = self.quant_method.apply(self, input_, bias)  显式的调用
                     > /workspace/global/vllm-hpu-extension/vllm_hpu_extension/awq_hpu.py(248)apply()
                         > weight = torch.ops.hpu.convert_from_uint4(qweight,…)
	• _initialize_kv_caches() 需要调用 model.forward()，因为 Transformer 结构只有在前向传播时才会创建 KV Cache。
	• 这样可以在推理前预分配 KV Cache，避免运行时分配导致的显存碎片化。
此步骤优化了 vllm 的高效推理，使得后续 continuous batching（连续批处理）和 PagedAttention（分页注意力）能够无缝利用缓存。
```

### LLM weights_load_device 参数 会反应在 
```
vllm/model_executor/model_loader/loader.py   load_config.device
       load_model()      load_config = vllm_config_load_config   是通过  vllm_config 里的 load_config 传递的
                          target_device = torch.device(load_device)
                          _initialize_model() -> get_model_architecture()  -> ./registry.py   ModelRegistry -> _VLLM_MODELS.items() -> DeepseekV2ForCausalLM
            _process_weights_after_loading
            -> module.process_weights_after_loading(model_config.dtype)
```

```
allow_async_output_proc 参数 gpu边生成 cpu边处理，减少最终返回结果的延迟
llm = LLM(model="meta-llama/Llama-2-7b-hf", allow_async_output_proc=True)
```

### Fused moe 调用过程
```
deepseek_v2.py 
DeepseekV2MoE
     self.gate = ReplicatedLinear()
     self.experts = FusedMoE()
     self.shared_experts = DeepseekV2MLP()
{
     router_logits = self.gate(hidden_states)
     self.experts(hidden_states, router_logits)
             
             topi_weight, topk_ids = select_experts()
                 grouped_topk
             torch.ops.hpu.mixture_of_experts   or   torch.ops.vllm.fused_marlin_moe()
}



FusedMoE.select_experts
   layer.py  select_experts()
       use_grouped_topk = true     grouped_topk()



Layer.py   FusedMoE 
    self.quant_method.create_weights(layer=self, **moe_quant_params)
    调用 weight_loader()   判断  quant_method channel block value，  下面有 weight_loader 调用 trace
```

Weight_loader 调用 trace
```
-> engine = cls(
  /workspace/global/vllm-fork/vllm/engine/llm_engine.py(273)__init__()
-> self.model_executor = executor_class(vllm_config=vllm_config, )
  /workspace/global/vllm-fork/vllm/executor/executor_base.py(262)__init__()
-> super().__init__(*args, **kwargs)
  /workspace/global/vllm-fork/vllm/executor/executor_base.py(51)__init__()
-> self._init_executor()
  /workspace/global/vllm-fork/vllm/executor/mp_distributed_executor.py(125)_init_executor()
-> self._run_workers("load_model",
  /workspace/global/vllm-fork/vllm/executor/mp_distributed_executor.py(189)_run_workers()
-> driver_worker_output = run_method(self.driver_worker, sent_method,
  /workspace/global/vllm-fork/vllm/utils.py(2293)run_method()
-> return func(*args, **kwargs)
  /workspace/global/vllm-fork/vllm/worker/hpu_worker.py(224)load_model()
-> self.model_runner.load_model()
  /workspace/global/vllm-fork/vllm/worker/hpu_model_runner.py(794)load_model()
-> self.model = get_model(vllm_config=self.vllm_config)
  /workspace/global/vllm-fork/vllm/model_executor/model_loader/__init__.py(14)get_model()
-> return loader.load_model(vllm_config=vllm_config)
  /workspace/global/vllm-fork/vllm/model_executor/model_loader/loader.py(445)load_model()
-> loaded_weights = model.load_weights(
  /workspace/global/vllm-fork/vllm/model_executor/models/deepseek_v2.py(846)load_weights()
-> weight_loader(param,
> /workspace/global/vllm-fork/vllm/model_executor/layers/fused_moe/layer.py(571)weight_loader()
```
