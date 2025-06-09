import torch

x = torch.randn(2,3,5)
print(x.size())
print(x)

x = x.permute(2,0,1)
print(x.size())
print(x)
