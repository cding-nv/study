import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = model.to('xpu')
data = data.to('xpu')
model = ipex.optimize(model)

# warm up
with torch.no_grad():
    model(data)
    model(data)
    model(data)

with profile(activities=[ProfilerActivity.XPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.no_grad():
            model(data)

print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))
