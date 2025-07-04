# llava 

Model: https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3    
Github code: https://github.com/haotian-liu/LLaVA      
 * Note:  Get the latest code. Commit: 237165b0166ad3fdc4fb0a7d881c857755d2404f 

## PVC
Code Patch:  [llava_bfloat16_gpu.patch](./llava_bfloat16_gpu.patch)

1. Steps:
 * Prepare ipex env
 * How to run in webserver:
   *  First terminal:
```
$ unset https_proxy
$ unset http_proxy
$ python -m llava.serve.controller --host 0.0.0.0 --port 10000
```
   *  Second terminal:
```
$ unset https_proxy
$ unset http_proxy
$ python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```
   *  Third terminal:
```
$ unset https_proxy
$ unset http_proxy
$ python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /public/llama_model/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
```

2.  It also can run in command line:
``` 
   $ python -m llava.eval.run_llava --model-path /public/llama_model/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ --image-file /home/fengding/angelababy.jpg --query "boy or girl?"
```
3.DeepSpeed by multi-cards (PVC, ARC770, Flex170) should also workable. (TODO)

4. Notes:
 * Bfloat16 can work correctly, but float16 output is gibberish.
 * Webserver output is very slow


## SPR


## DeepSpeed (PVC, flex170, arc770) (TODO)
