$ python test_softmax_bf16.py

```
fp32 tensor:  tensor([2542.3423, 2545.3633, 2541.4182, 2547.9465])
     softmax result:  tensor([0.0034, 0.0699, 0.0014, 0.9254])
to be bf16:  tensor([2544., 2544., 2544., 2544.], dtype=torch.bfloat16)
     softmax result:  tensor([0.2500, 0.2500, 0.2500, 0.2500], dtype=torch.bfloat16)
fp32 to be norm:  tensor([-5.6042, -2.5833, -6.5283,  0.0000])
     softmax result:  tensor([0.0034, 0.0699, 0.0014, 0.9254])
input_norm_bf16  tensor([-5.5938, -2.5781, -6.5312,  0.0000], dtype=torch.bfloat16)
     softmax result:  tensor([0.0034, 0.0703, 0.0014, 0.9258], dtype=torch.bfloat16)
```
