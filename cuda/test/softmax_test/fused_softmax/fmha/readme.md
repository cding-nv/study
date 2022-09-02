# 功能说明
- 融合了SwinTransformer中MHA的qk_result + relative_position_bias + mask + softmax部分。
- 在原python代码中有dropout，但是官方的训练和测试的部分该参数均为0，因此，目前删去了dropout相关的部分功能。这部分也是考虑到如果使用dropout，需要额外生成用于dropout部分的random tensor并传入dropout rate。这部分最初已开发并测试好，因为上述原因目前这一版已删去。如果dropout部分实际激活，预计会有更高的性能收益。
- 融合之后的forward与backward部分，每次均可以比原始代码减少1-2个kernel调用。
- 在A100-80G单卡batch_size=192，fp16模式下，结合其他已有加速，目前达到325image/second的训练速度，该部分的在原有基础上大约带来1%性能提升。
- 由于测试模块的序列长度为49，因此无法在fp16下使用half2类型，无法享受half2带来的进一步提升。


# 使用说明

1. 在代码目录下进行安装   
`python setup install --user`

- 安装完成后，可以使用`python test.py`进行单元测试。fp16模式下的精度测试中，forward/backward部分使用了差值与`1.22e-4`的比较，平均差异在`1e-5`的数量级。1.22e-4由fp32与fp16的指数部分的精度差值13bit得到，1/2^13。

2. 在SwinTransformer的模型定义文件（swin_transformer.py）中添加融合模块（代码如下）

```python
class MySoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len):
        if attn_mask is not None:
            softmax_cuda.softmax_fwd(input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len)
        else:
            softmax_cuda.softmax_nomask_fwd(input, relative_pos_bias, batch_size, window_num, num_head, window_len)
        
        ctx.save_for_backward(input)
        ctx.batch_size = batch_size
        ctx.window_num = window_num
        ctx.num_head = num_head
        ctx.window_len = window_len
        return input

    @staticmethod
    def backward(ctx, grad_out):
        softmax_result = ctx.saved_tensors[0]
        batch_size = ctx.batch_size
        window_num = ctx.window_num
        num_head = ctx.num_head
        window_len = ctx.window_len

        softmax_cuda.softmax_bwd(
            grad_out.contiguous(), softmax_result, batch_size, window_num, num_head, window_len)

        return grad_out, torch.sum(grad_out, dim=0), None, None, None, None, None
```

3. 在WindowAttention类中添加对应的参数`batch_size`，并在对应位置进行该模块的调用即可。

```python
relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

'''
# original code
attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N)+ mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)
    attn = self.attn_drop(attn)
'''
# use fused kernel
attn = MySoftmax.apply(attn, relative_position_bias, mask, self.batch_size, B_ // self.batch_size, self.num_heads, N)

x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
```