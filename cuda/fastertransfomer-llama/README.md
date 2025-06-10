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

## Run
```
$ ./bin/llama_example
```
