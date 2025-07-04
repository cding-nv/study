# vllm on Gaudi

## Env Setup
```
sudo docker run -it -d --name=fengding --runtime=habana  -v /home/jenkins/fengding:/home/fengding -e HABANA_VISIBLE_DEVICES=7 -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest /bin/bash

export https_proxy=http://proxy.ims.intel.com:911
export http_proxy=http://proxy.ims.intel.com:911

## Install vllm-hpu-extension, neural_compressor and vllm
git clone https://github.com/HabanaAI/vllm-fork.git
pip install -r requirements-hpu.txt
python setup.py develop --user

## Check
$ pip list |grep vllm
vllm                              0.6.3.dev1122+g2f43ebf5.d20241121.gaudi118 /home/fengding/vllm-fork
vllm-hpu-extension                0.1

$ VLLM_SKIP_WARMUP=true python3 examples/offline_inference.py
```
## Run fp8 calibration
```
git clone https://github.com/HabanaAI/vllm-hpu-extension
cd vllm-hpu-extension/calibration

# For Llama-3.1-8B
./calibrate_model.sh -m /home/fengding/Llama-3.1-8B/ -d /home/fengding/processed-data.pkl -o ./output -b 128 -t 1 -l 128
     ## Generate scale factors in ./output

# For Llama-3.1.8B-Instruct
./calibrate_model.sh -m /home/fengding/Llama-3.1-8B-Instruct/ -d /home/fengding/processed-data.pkl -o ./output_llama3.1.8b.Instruct -b 128 -t 1 -l 128
    ## Generate scale factors in ./output_llama3.1.8b.Instruct
```

## Start vllm server
```
cd vllm-fork/

PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_CONTIGUOUS_PA=true \
VLLM_SKIP_WARMUP=true \
QUANT_CONFIG=/home/fengding/vllm-hpu-extension/calibration/output/output_llama3.1.8b.Instruct/maxabs_quant_g2.json \
python3 -m vllm.entrypoints.openai.api_server \
--model /home/fengding/Llama-3.1-8B-Instruct/ \
--port 8080 \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size 1 \
--disable-log-requests \
--block-size 128 \
--quantization inc \
--kv-cache-dtype fp8_inc \
--device hpu \
--weights-load-device cpu \
--dtype bfloat16 \
--num_scheduler_steps 16 2>&1 > vllm_serving.log &
```

## Start Client to test
```
curl --noproxy "*" http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{"model": "/home/fengding/Llama-3.1-8B-Instruct/", "prompt": "San Francisco is a", "max_tokens": 100}'
```

## Run Benchmark
```
python benchmarks/benchmark_serving.py \
--backend vllm \
--model /home/fengding/Llama-3.1-8B-Instruct  \
--dataset-name sonnet \
--dataset-path benchmarks/sonnet.txt \
--request-rate 128 \
--num-prompts 128 \
--port 8080 \
--sonnet-input-len 128 \
--sonnet-output-len 128 \
--sonnet-prefix-len 100
```

## Code Structure
<img align="center" src="./vllm_gaudi.png" width="100%" height="100%">

## Reference
https://github.com/HabanaAI/vllm-fork    
https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#quantization-fp8-inference-and-model-calibration-process    
https://github.com/HabanaAI/vllm-hpu-extension    

