from diffusers import StableDiffusionXLPipeline
import torch
import time
import os
import intel_extension_for_pytorch as ipex

os.environ['OCL_ICD_VENDORS'] = '/etc/OpenCL/vendors'

# from https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# or https://www.modelscope.cn/AI-ModelScope/stable-diffusion-xl-base-1.0.git
model_path = r"path/to/stable-diffusion-xl-base-1.0"

device = "xpu"
amp_dtype = torch.bfloat16

pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=amp_dtype, use_safetensors=True)
pipeline.to(device)

prompt = "A beautiful landscape with mountains and a lake"

start_time = time.time()

with torch.amp.autocast(device_type=device, dtype=amp_dtype):
    image = pipeline(prompt=prompt, generator=torch.manual_seed(33)).images[0]

print(f"Time taken: {time.time() - start_time} seconds")
image.save("output_op.png")


start_time = time.time()

from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.XPU]) as prof:
    with torch.amp.autocast(device_type=device, dtype=amp_dtype):
        image = pipeline(prompt=prompt, generator=torch.manual_seed(33)).images[0]

print(f"Time taken: {time.time() - start_time} seconds")
image.save("output_op.png")

print(prof.key_averages().table(row_limit=-1))

torch.save(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1), "./profile.pt")
