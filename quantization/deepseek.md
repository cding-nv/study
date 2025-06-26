# Notes

####
1. Fp8 训练   正反都是用 E4M3  并通过细粒度的per-tile（1x128）和per-group（128x128）量化来降低误差，  DeepSeek-V3展示了per-tile和per-group量化对于模型收敛的重要性    
      attention 后的 linear 输入的精度需要提升这样的细节，以及独立实现 per-group scaling 的训练   ??    
2. deepseek 提出的将 pretrain batch size 从传统的 4M~8M tokens，提升至 4K * 15360 = 60M tokens 就是这样的变化。超大的 batch size 可能可以 makes pipeline parallelgreat again。  Pp+dp    
3. MLA  + MOE    
	MLA 为了降低 kv cache/token 的开销。  类似 LoRA 的方式对 kv 降维压缩，同时升维操作转移到 Q 和 O    
	MOE 为了降低 flops/param 的开销  v3 采用小专家设计  256 专家，总参数671B，激活参数量仅增加 37B    
DeepSeek-V3可以将通信计算比例控制在大约1:1，这为后面的通信隐藏带来了机会    

DeepSeek V3   是MOE 模型，专注通用基础架构和预训练权重    
DeepSeek-R1    Reasoning1, 专注推理的第一个版本。 是通过 v3 通过强化学习 RL 和 蒸馏技术提升推理能力， 在 RL基础上引入冷启动数据和 SFT    
Deepseek-R1-zero,   通过强化学习RL 直接优化基础模型，无监督微调 SFT    
DeepSeek-R1-Distill-Qwen-32B  蒸馏模型    

### SRAM
容量大的SRAM访问时间就越长，同样制程和设计的情况下，访问延时与容量的开方大致是成正比的.针对L1/L2的不同角色，L1更加注重速度， L2更加注重节能和容量    
由于L3 Cache的时延要求没有那么高，现在大家也要考虑不使用SRAM，转而使用STT-MRAM，或是eDRAM来做L3 Cache。    
不用SRAM主要原因是因为很难集成更大容量。一个SRAM cell占用4-6transistor ，而eDRAM和STTRAM 只需要1-2个，所以能比SRAM有更大容量。eDRAM其实并不比SRAM慢，用它替代SRAM是为了增大容量。用STT RAM是不仅能增大容量，还能减少静态能耗    

Huawei 昇腾，  Cache & Buffer: SOC片内有层次化的memory结构， AI core内部有两级的menory buffer, SOC片上还有8MB L2的buffer, 专门用来AI Core， AI CPU, 提供高带宽，低延迟的memory结构， 芯片内部还继承了LPDDR4x控制器，为芯片提供更大容量的DDR内存    

寄存器 具有高性能、高耗电的特点，所以造价成本较高。是距离CPU最近、访问数据速度最快的存储介质，用来做一些最基本的指令和地址存储。寄存器不需要很大容量，但是必须有匹配CPU的数据传输速度    
DRAM：   dynamic random-access memory   易失 大容量DRAM 之所以叫动态，是因为将数据写入 DRAM 后，一段时间过后数据会丢失，需要一个额外的电路不断对其进行刷新操作才行。    
      因为    DRAM储存数据利用的是电容中是否有电荷，有代表1，无代表0。但是电容会放电或吸电，刷新操作会对其进行检查。如果电量大于满电量的1/2，则将电充满，否则将电全部放掉    
      缓存使用的是比一般的RAM(Random Access Memory)存取速度更快的SRAM(Static Random Access Memory)    
      RAM一般分为静态RAM(static RAM, SRAM)和动态RAM(dynamic RAM, DRAM)。SRAM 的速度虽然比 DRAM 快，但成本高得多，所以不可能将 DRAM 全部替换成 SRAM。    
      CPU访问数据先是在一   级缓存(L1 Cache)中找，找不到再到二级缓存(L2 Cache)中找，再没有就去内存中找。    

缓存有以下特点：    
    更高速访问：缓存的访问速度比内存更快，因为它通常使用更快的存储技术(如SRAM)。    
    小容量：缓存的容量通常较小，常见的以KB或MB为单位。    
    多级结构：现代计算机系统中，缓存通常有多级结构，如L1、L2和L3缓存。L1缓存是最快的，位于处理器核心内部；L2缓存稍慢，容量更大；L3缓存更慢，但容量更大，通常在整个处理器芯片上共享    

静态RAM，SRAM(Static Random-Access Memory)    
    优点：访问速度快、功耗低、数据稳定(只要有店员，数据就不会丢失)    
    缺点：存储密度低(集成度低，存储相同的数据，SRAM 的体积是 DRAM 的 6 66 倍)、成本高(同容量的 SRAM 是 DRAM 的 4 44 倍)      
    用途：主要用于 CPU 缓存，也可用在某些嵌入式系统中作为主内存    
动态RAM，DRAM(Dynamic Random-Access Memory)    
    优点：存储密度高、成本低    
    缺点：访问速度较慢，需要周期性刷新以保持数据。    
    用途：主要用作计算机的主内存。    


