# Run vllm on Nvidia GPU
## Docker run
```
docker run -ti -d  --env CUDA_VISIBLE_DEVICES=3 --gpus all --ipc=hosty -e  https_proxy=$https_proxy -e no_proxy=$no_proxy  -v /home/test/fengding:/home/fengding/ --name lock=-1 --rm nvcr.io/nvidia/pytorch:23.10-py3
```
## Install vllm
```
pip install vllm==0.7.1
```

## Error handling
1.  AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
```
# Comment this line
/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py  162   
      #LayerId = cv2.dnn.DictValue
```   
2. RuntimeError("Failed to infer device type")    
```
   # Specify the device to be 'cuda'
   llm = LLM(model="/home/fengding/Qwen2-7B-Instruct", device='cuda')
```
