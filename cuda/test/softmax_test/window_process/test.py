import torch
import swin_window_process
import random
import unittest


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pyt_forward(x, shift_size, window_size):
    # x in shape(B, H, W, C)
    # cyclic shift
    if shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x
    # partition windows
    x_windows = window_partition(shifted_x, window_size)
    return x_windows


def reverse_pyt_forward(attn_windows, shift_size, window_size, H, W):
    # x in shape(B*nH*nW, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)
    if shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x
    return x


def copy_one_tensor(input, requires_grad=True):
    input1 = input.clone().detach().requires_grad_(requires_grad).cuda()
    return input1

class Test_WindowProcess(unittest.TestCase):
    def setUp(self):
        self.B = 192
        self.H = 56
        self.W = 56
        self.C = 96
        self.shift_size = 2
        self.window_size = 7
        #self.dtype = torch.float16
        self.nH = self.H // self.window_size
        self.nW = self.W // self.window_size
    
    def test_roll_and_window_partition_forward(self, dtype=torch.float32):
        input = torch.randn((self.B, self.H, self.W, self.C), dtype=dtype, requires_grad=True).cuda()
        
        input1 = copy_one_tensor(input, True)
        input2 = copy_one_tensor(input, True)

        with torch.no_grad():
            # ori
            expected = pyt_forward(input1, self.shift_size, self.window_size)
            # my kernel
            my_output = WindowProcess.apply(input2, self.B, self.H, self.W, self.C, -self.shift_size, self.window_size)
        
        self.assertTrue(torch.allclose(expected, my_output, rtol=1e-05, atol=1e-08))
    
    def test_roll_and_window_partition_backward(self, dtype=torch.float32):
        input = torch.randn((self.B, self.H, self.W, self.C), dtype=dtype, requires_grad=True).cuda()
        d_loss_tensor = torch.randn((self.B*self.nW*self.nH, self.window_size, self.window_size, self.C), dtype=dtype).cuda()
        
        input1 = copy_one_tensor(input, True)
        input2 = copy_one_tensor(input, True)

        # ori
        expected = pyt_forward(input1, self.shift_size, self.window_size)
        expected.backward(d_loss_tensor)
        # my kernel
        my_output = WindowProcess.apply(input2, self.B, self.H, self.W, self.C, -self.shift_size, self.window_size)
        my_output.backward(d_loss_tensor)
        
        self.assertTrue(torch.allclose(expected, my_output, rtol=1e-05, atol=1e-08))

    def test_window_merge_and_roll_forward(self, dtype=torch.float32):
        input = torch.randn((self.B*self.nH*self.nW, self.window_size, self.window_size, self.C), dtype=dtype, requires_grad=True).cuda()
        
        input1 = copy_one_tensor(input, True)
        input2 = copy_one_tensor(input, True)

        with torch.no_grad():
            # ori
            expected = reverse_pyt_forward(input1, self.shift_size, self.window_size, self.H, self.W)
            # my kernel
            my_output = WindowProcessReverse.apply(input2, self.B, self.H, self.W, self.C, self.shift_size, self.window_size)
        
        self.assertTrue(torch.allclose(expected, my_output, rtol=1e-05, atol=1e-08))
    

    def test_window_merge_and_roll_backward(self, dtype=torch.float32):
        input = torch.randn((self.B*self.nH*self.nW, self.window_size, self.window_size, self.C), dtype=dtype, requires_grad=True).cuda()
        d_loss_tensor = torch.randn((self.B, self.H, self.W, self.C), dtype=dtype, requires_grad=True).cuda()
        
        input1 = copy_one_tensor(input, True)
        input2 = copy_one_tensor(input, True)

       # ori
        expected = reverse_pyt_forward(input1, self.shift_size, self.window_size, self.H, self.W)
        expected.backward(d_loss_tensor)
        # my kernel
        my_output = WindowProcessReverse.apply(input2, self.B, self.H, self.W, self.C, self.shift_size, self.window_size)
        my_output.backward(d_loss_tensor)
        
        self.assertTrue(torch.allclose(expected, my_output, rtol=1e-05, atol=1e-08))

    def test_roll_and_window_partition_forward_fp16(self, dtype=torch.float16):
        self.test_roll_and_window_partition_forward(dtype=dtype)

    def test_roll_and_window_partition_backward_fp16(self, dtype=torch.float16):
        self.test_roll_and_window_partition_backward(dtype=dtype)

    def test_window_merge_and_roll_forward_fp16(self, dtype=torch.float16):
        self.test_window_merge_and_roll_forward(dtype=dtype)
    
    def test_window_merge_and_roll_backward_fp16(self, dtype=torch.float16):
        self.test_window_merge_and_roll_backward(dtype=dtype)

if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main(verbosity=2)
