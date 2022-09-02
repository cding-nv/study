# 功能说明
融合了SwinTransformer中关于shift window及window划分的操作，包括
- window cyclic shift和window partition
- window merge和reverse cyclic shift

在使用window shift的block中，原始的shift操作使用torch.roll需要调用2个cuda kernel以及window partition/window merge，总计3个cuda kernel。本身这部分的所有变换都类似reshape/view的操作，融合之后只需1个cuda kernel。对于不shift的情况，本身只使用了1个cuda kernel，因此不做变动。
torch.roll原始kernel运行时间占比为4.6%，融合后无torch.roll相关kernel。

使用kernel之后，在A100-80G上测试，训练速度由原来的325 image/s可以提升至339 image/s。


# 使用说明

1. 在代码目录下进行安装   
`python setup install --user`


2. 在SwinTransformer的模型定义文件（swin_transformer.py）中添加融合模块（代码如下）

```python
import swin_window_process

class WindowProcess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = swin_window_process.roll_and_window_partition_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size
        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = swin_window_process.roll_and_window_partition_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = swin_window_process.window_merge_and_roll_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size

        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        #grad_out = ctx.saved_tensors[0]
        #grad_out = torch.zeros((B, H, W, C), dtype=dtype).cuda()
        grad_out = swin_window_process.window_merge_and_roll_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None
```

3. 在SwinTransformerBlock类的forward函数中的对应部分，完成替换。
（这部分原始代码的window_partition和window_reverse在外部，为了方便我稍稍改了一下。）

```python
# cyclic shift & partition windows
if self.shift_size > 0:
    # original code
    # shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    # x_windows = window_partition(shifted_x, self.window_size)

    x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
else:
    shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)

x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
...
```

```python
# merge windows & reverse cyclic shift
if self.shift_size > 0:
    # original code
    # shifted_x = window_reverse(attn_windows, self.window_size, H, W)
    # x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

    x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
else:
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    x = shifted_x

x = x.view(B, H * W, C)
...
```