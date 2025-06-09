# Deepseek 

## 1. Run on CPU

```
docker run --rm -it -d --privileged -v /home/fengding/:/home/fengding intel/intel-extension-for-pytorch:2.4.0-pip-base

pip install triton==3.0.0
pip install transformers==4.46.3
pip install safetensors==0.4.5

python model.py
    output: torch.Size([2, 102400])
```

## 2. Deepseek MLA, naive vs absorb
<img align="center" src="./deepseek-mla.png" width="100%" height="100%">

## 3. Deepseek layer, moe
<img align="center" src="./deepseek-layer-moe.png" width="100%" height="100%">

## 4. Deepseek R1 MLP, MOE
<img align="center" src="./deepseek-r1-mlp-moe.png" width="100%" height="100%">