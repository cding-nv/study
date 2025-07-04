# Get dtype

# How to run on CPU
## Prepare torch env
```
docker run --rm -it -d --privileged -v /home/fengding/:/home/fengding intel/intel-extension-for-pytorch:2.4.0-pip-base
```
## Run
```
$ python get_dtype.py

Softmax input dtype: torch.bfloat16
Softmax output dtype: torch.bfloat16
Output dtype: torch.bfloat16
Layer: , Type: SimpleAttention, Data Type: torch.float8_e4m3fn
Layer: query, Type: Linear, Data Type: torch.float8_e4m3fn
Layer: key, Type: Linear, Data Type: torch.float8_e4m3fn
Layer: value, Type: Linear, Data Type: torch.float8_e4m3fn

```
 
## How to print dtype
### 1. From module.parameters()
```
for name, module in model.named_modules():
    if len(list(module.parameters())) > 0:
        print(f"Layer: {name}, Type: {module.__class__.__name__}, Data Type: {next(module.parameters()).dtype}")
```
Code trace
```Python
torch/nn/modules/module.py
    parameters()
      named_parameters()
        self._named_members()   module._parameters.items()

self._parameters[name]
    def register_parameter(self, name: str, param: Optional[Parameter])  -> self._parameters[name] = param
```
There is 2 methods to add parameters
* nn.Parameter() in __init__()
* register_parameter()
### 2. Register hook
attn_model.softmax.register_forward_pre_hook(print_softmax_input_dtype)    
attn_model.softmax.register_forward_hook(print_softmax_output_dtype)
### 3. Add print before/after op in forward
For example, torch.matmul(), torch.einsum()