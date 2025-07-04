import torch

m = torch.nn.Softmax(dim=0)

input_fp32 = torch.tensor([2542.34228515625, 2545.36328125, 2541.418212890625, 2547.946533203125])
print("fp32 tensor: ", input_fp32)
output = m(input_fp32)
print("     softmax result: ", output)

input_bf16 = input_fp32.to(torch.bfloat16)
print("to be bf16: ", input_bf16)
output = m(input_bf16)
print("     softmax result: ", output)

input_norm = torch.tensor([-5.604248046875, -2.583251953125, -6.5283203125, 0.0])
print("fp32 to be norm: ", input_norm)
output = m(input_norm)
print("     softmax result: ", output)

input_norm_bf16 = input_norm.to(torch.bfloat16)
print("input_norm_bf16 ", input_norm_bf16)
output = m(input_norm_bf16)
print("     softmax result: ", output)
