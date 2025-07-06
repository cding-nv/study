

cuda/vanilla/src/multi_head_attention.cu    
  softmax_kernel_v3    
       blockReduceMax    
       blockReduceSum    
       warp 里用 __shfl_xor_sync 累加 或者 找最大值， warp id 为下标存在 shared mem,   
       赋值给 blockDim/32 个数个 thread ，再用 warp __shfl_xor_sync 累加 或者 找最大值
         如果 blockDim/32 > 32 怎么办呢？   block 的 threads 不会大于 1024 ？    
  最后有 dropout_rate 的计算， 有mask ， 最开始有 scale



在  cuda/vanilla/sample/__multi_head_attention_grad.py  有 注册反向    前向其实只有 mha 的一部分 但 register 成 MultiHeadAttention    
     @ops.RegisterGradient("MultiHeadAttention")    
     _multi_head_attention_grad_cc

    grad_op.h  -> SoftmaxGrad  -> backward()
    softmax_grad.cu -> SoftmaxGrad()    
    general_kernels.cu -> launch_backprop_masking_softmax -> backprop_masking_softmax    
