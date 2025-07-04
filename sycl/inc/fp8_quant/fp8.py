import torch

def quantize_fp8_e4m3(weight):
    """将 FP32 权重量化到 FP8 E4M3 格式"""
    # 1. 计算权重最大绝对值
    max_abs = torch.max(torch.abs(weight))

    # 2. 计算 FP8 量化 scale（缩放因子）
    scale = max_abs / (2 ** 6)  # FP8 动态范围

    # 3. 计算 weight_scale_inv（反量化因子）
    weight_scale_inv = 1.0 / scale

    # 4. 量化权重（转换到 FP8 近似表示）
    weight_quant = torch.clamp(torch.round(weight * weight_scale_inv), -127, 127).to(torch.int8)

    return weight_quant, weight_scale_inv

def dequantize_fp8_e4m3(weight_quant, weight_scale_inv):
    """将 FP8 量化数据恢复到 FP32"""
    return weight_quant.float() * weight_scale_inv

class FP8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化 FP32 权重
        self.weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32)

        # 进行 FP8 量化
        self.weight_quant, self.weight_scale_inv = quantize_fp8_e4m3(self.weight_fp32)

    def forward(self, x):
        """前向传播：使用量化权重进行计算"""
        # 反量化权重
        weight_dequant = dequantize_fp8_e4m3(self.weight_quant, self.weight_scale_inv)

        print("No quant: ", torch.matmul(x, self.weight_fp32.T))
        # 计算 x @ W
        return torch.matmul(x, weight_dequant.T)

# 测试 FP8 线性层
fp8_linear = FP8Linear(in_features=4, out_features=3)
x = torch.randn(2, 4, dtype=torch.float32)
output = fp8_linear(x)

print("Input:\n", x)
print("Output:\n", output)
