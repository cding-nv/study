from neural_compressor.torch.quantization import (
    FP8Config,
    prepare,
    convert,
)

import torch
import torchvision.models as models

model = models.resnet18()
qconfig = FP8Config(fp8_config="E4M3")
model = prepare(model, qconfig)

# customer defined calibration
model(torch.randn(1, 3, 224, 224).to("hpu"))

model = convert(model)

output = model(torch.randn(1, 3, 224, 224).to("hpu")).to("cpu")
print(output.shape)

