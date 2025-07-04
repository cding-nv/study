import os
import sys

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

model_path = sys.argv[1]
print("#### model_path", model_path)

#dirname = sys.argv[2]
#os.makedirs(dirname, exist_ok=True)

with safe_open(model_path, framework="pt", device='cpu') as f:
    for key in f.keys():
        value = f.get_tensor(key)
        print(key, value.size(), value.dtype, type(value))
        #print(key, value.max(), value.size(), value.dtype, type(value))

#        if isinstance(value, torch.Tensor):
#            print(value.shape, value.dtype)
#            with open(os.path.join(dirname, key + '.bin'), 'wb') as ff:
#                if value.dtype == torch.bfloat16:
#                    value = value.to(torch.float16)
#                ff.write(value.cpu().numpy().tobytes())
