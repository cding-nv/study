# Video-llava, run on PVC by IPEX

Github: https://github.com/PKU-YuanGroup/Video-LLaVA.git    
Model: https://huggingface.co/LanguageBind/Video-LLaVA-7B 


## Steps

```
Env
# python 3.10
# source oneapi 2024.0
# Install IPEX

$ pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
$ pip install transformers==4.33.0
$ pip install einops peft sentencepiece protobuf
```

## Code change
```
$ git clone https://github.com/PKU-YuanGroup/Video-LLaVA.git
$ git clone https://huggingface.co/LanguageBind/Video-LLaVA-7B 

$ cd Video-LLaVA
$ git apply videollava_to_xpu.patch
         # Commit HEAD: a8404b991f0ca186ea5071663dbeb19ff1352fbb
```

## Run
```
For image
$ python -m videollava.serve.cli --model-path Your_Path_Video-LLaVA-7B/ --file ./videollava/serve/examples/waterview.jpg --device xpu

For video
$  python -m videollava.serve.cli --model-path Your_Path_Video-LLaVA-7B/ --file ./videollava/serve/examples/sample_demo_1.mp4 --device xpu
```
