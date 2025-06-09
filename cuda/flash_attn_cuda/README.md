# Run

python setup.py clean
python setup.py build_ext --inplace
python test_flash_attn.py

# Notes

![](./flash-attn-algo.png)
![](./flash-attn-loop.png)

1. https://fancyerii.github.io/2023/10/23/flashattention/  论文解读    
2. M is SRAM size. N is the sequence length and d is the head dimention. For GPT2, N=1024, d = 64    
3. 外循环是 K 和 V 移到下一个 block    
4. 内循环是 Qi, 一个一个的， 去和  K的某一个Kj block， V的某一个 Vj block，去乘    
5. 第12 行是关键，  局部的  Pij  （相当于求softmax的分子部分）就可以和 V 提前相乘了, 多少个 j ， 就要和 V 乘多少次， 而原始的完整的 softmax结果和 V的一列只乘一次. 乘的次数变多，但每次 Pij 的长度变短了    
6. 	flash attention 明显对  batch/training 即多query 是有收益的，通过减少 hbm 的读写访问，那么对 inference next token, 有帮助吗， 即每次就一个 query，？    
7. 一般情况下， 一个 token， 和先前所有token （包括它自己）算attention， 把这个token 的所有 sotfmax 都算出来再乘V， SRAM 应该是放的下的吧。 除非历史token 太多 （K V 太多）， SRAM 装不下，那么用 flash attention 才有收益对么？
8. Flash attention-2 主要就是 把 query 横着切了一下 ？ 之前 flash attention-1 是 所有 token Qi 一起计算的？    
9. 算法感知 IO， 将内存复杂度从O(N²)降低到O(N)    
10. https://pytorch.org/blog/flash-decoding/    flash-decoding 又把  K， V 切了几刀， 相当于外循环并行起来了~~    

11. https://zhuanlan.zhihu.com/p/607364156  on Intel GPU    

12. https://crfm.stanford.edu/2023/10/12/flashdecoding.html Flash-Decoding for long-context inference    
