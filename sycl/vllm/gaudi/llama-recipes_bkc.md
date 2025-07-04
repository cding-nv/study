This page is BKC of MMLU_Pro dataset accuracy benchmark on vllm.
Reference page: [meta_eval_reproduce](https://github.com/meta-llama/llama-recipes/tree/b5f64c0b69d7ff85ec186d964c6c557d55025969/tools/benchmarks/llm_eval_harness/meta_eval_reproduce)

## Prepare Code 
Cherry-pick 2 PRs to fix [issue](https://github.com/meta-llama/llama-recipes/issues/674) [PR-676](https://github.com/meta-llama/llama-recipes/pull/676)

```
git clone https://github.com/meta-llama/llama-recipes.git
git checkout b5f64c0b69d7ff85ec186d964c6c557d55025969 -b br_b5f64c0

git pull
git fetch origin pull/676/head:PR_676
git cherry-pick 94c637418967fbf9e3fe6d2bd744ff5ff31efc6c
git cherry-pick 30392de3844ff77431fce8277a61bffe0ed71ad4
```
## Prepare vllm docker
## Env
```
export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
export HF_ENDPOINT=https://hf-mirror.com
 
# My huggingface token: 
export HUGGINGFACEHUB_API_TOKEN=
h
f
_
QQJzV
jiLQrn
OowKT
DBJtz
GweVwCefPCbYw
pip install huggingface_hub
huggingface-cli login

pip install lm-eval[math,ifeval,sentencepiece,vllm]==0.4.7

pip uninstall typing_extensions
pip uninstall openai
pip install typing_extensions==4.11.0
pip install openai==1.53.0
```
Remove model name prefix “Meta-“ in [line](https://github.com/meta-llama/llama-cookbook/blob/b5f64c0b69d7ff85ec186d964c6c557d55025969/tools/benchmarks/llm_eval_harness/meta_eval_reproduce/prepare_meta_eval.py#L30)        -> str = str[len('Meta-'):]    
Remove self.DATASET_NAME prefix “Meta-” in /usr/local/lib/python3.10/dist-packages/lm_eval/api/task.py:930    
Notes: Dateaset lighteval/MATH-Hard has been [404](https://huggingface.co/datasets/lighteval/MATH-Hard)

## Run
```
cd tools/benchmarks/llm_eval_harness/meta_eval_reproduce
# Modify eval_config.yaml, change “tasks:” 
#       tasks: "meta_mmlu_pro_instruct"
# Change tensor_parallel_size, data_parallel_size to be 1 for single device.

python prepare_meta_eval.py --config_path ./eval_config.yaml
# Print command: 
# lm_eval --model vllm   --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=8192,add_bos_token=True,seed=42 --tasks meta_mmlu_pro_instruct --batch_size auto --output_path eval_results --include_path /home/fengding/llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval_reproduce/work_dir --seed 42  --log_samples
```
Run the above print command, and get result,
|Tasks|Version|Filter|n-shot|Metric|Value|Stderr|
|------|-------|------|------|------|-----|------|
|meta_mmlu_pro_instruct|1|strict-match|0|exact_match|0.4682|+- 0.0045| 

### lm_eval[vllm] calling stack
```
/usr/local/bin/lm_eval  -> cli_evaluate()
lm_eval/__main__.py -> evaluator.simple_evaluate()
lm_eval/evaluator.py  -> simple_evaluate()
lm_eval/api/model.py  -> create_from_arg_string()
lm_eval/models/vllm_causallms.py ->
         class VLLM(TemplateLM):
                self.model = LLM(**self.model_args)
                _model_generate() -> self.model.generate()                                  
```

## Reference
n-shot, COT:  https://www.promptingguide.ai/techniques/fewshot
