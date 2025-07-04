# How to run stable diffusion XL on ipex-gpu

## Steps:

Model:     
$ git clone https://www.modelscope.cn/AI-ModelScope/stable-diffusion-xl-base-1.0.git    

```
$ docker run -it  --device=/dev/dri --ipc=host --net=host -e http_proxy=$http_proxy -e https_proxy=https_proxy intel/intel-extension-for-pytorch:2.3.110-xpu /bin/bash    
$ pip install diffusers transformers einops    
$ python test_xl.py    
```
