# SYCL 编程模型

1. sycl 编程模型，跨异构平台， 对应语言是  dpc++， 标准 c++ 语言。（也有dpc++编程教程）
采用的是 LLVM 开源框架，实现了一些 sycl api，runtime， 编译驱动，前端 clang 编译工具，memory/device 管理, 多流管理，支持 JIT，AOT 等等
不同的 device 如果要支持 DPC++ 的框架需要将device 的编译工具链 和 sycl 工具链的整合， 还有 device runtime 和 sycl runtime 的整合，这部分按照 plugin inferface 实现对应的 runtime 和 driver，以动态库dlopen方式加载。    
https://github.com/intel/llvm/blob/sycl/sycl/doc/design/PluginInterface.md    

2. https://codeplay.com/solutions/oneapi/for-cuda/  这个 codeplay 公司实现了 dpc++/sycl 工具链和 nvidia 编译工具链 还有 ptx 的对接，对接上后， intel 的 oneDNN 算子库， oneMKL 数学库也能在 nvidia 的 GPU 上跑。 目前 codeplay 被 intel 收购了    

3. https://www.oneapi.io/event-sessions/enable-amd-gpu-for-dpc/ 这个是 中科院计算所实现了 dpc++/sycl 和 AMD GPU 的对接    

4.  cuda code 转 dpc++， 用的方法就是 用 clang 前端转 AST

5. 超标量 -> SIMT -> DSA  https://zhuanlan.zhihu.com/p/387269513 
   * 超标量（乱序、预取、分支预测、...）、并行（向量、SIMT、多核、...）、专用（DSA）
   * DSA路线最大的困难在软件。只要想吃DSA的收益，一定会带来软件上的巨大困难。吃的越多困难越大，吃得少一点也会引入很大的困难，除非只吃一次，当然只吃一次收益也就只有一次。即使像NV这样拥有强健cuda生态的公司，引入TensorCore一样需要承受这样的痛苦。cuda只能覆盖SIMT的软件问题，覆盖不了TensorCore，同样需要能解决DSA路线困境的软件方案。而以DSA为主的AI芯片，则更是会感受到软件上的巨大痛苦。

   * DSA指令的性能来源于固化了粗粒度的一整块计算，这些接口是需要通过ISA直接暴露给软件的，软件的主要痛苦则是来源于ISA的高度碎片化和不稳定性。不稳定性是软件最大的障碍，给一款芯片做适配，即使工程量再大，逼近芯片都固定了，总归是可以适配下来的。但不稳定性意味着适配的工程量无法被下一代芯片复用，那此时的适配工程量就是无穷无尽的了。应用是没法跟着芯片的迭代周期进行迭代的，生态就很难建立起来，于是很多人就诉诸编译器。

   * 编译器不是万能的。一定程度上很多架构师都喜欢画大饼用一个编译器衔接上下各种碎片化的前端和后端，似乎这样整个架构图就一切都舒爽了，啥都能自由支持，而且自动优化。但实际上编译器的能力是非常有限的，编译器的自动化能力很大程度依赖于IR的完备性，也就是图灵祖师爷给我们奠定的超强基础。而编译器的优化能力嘛，且不说AI编译器这种才几年探索的新兴事物，HPC领域卷了几十年了，要是编译器优化能力可以依赖那还有blas库啥事呢？而这两点其实在AI这一领域更是崩坏，所以这种架构图虽然吹牛一时爽，最后编译器其实是有多少人工就有多少自动化。
   * tiling影响的则是体系结构最核心的瓶颈，也就是缓存的利用
   * 体系结构核心瓶颈是访存，因此调度的核心在于优化访存。访存一方面取决于代码执行顺序，一方面取决于片上缓存如何分配
   * Halide/TVM 核心在于解耦计算和调度，可以复用的还是计算，但计算本身的工程量很小，真正的大头还是调度，调度还是极其碎片化的。调度不仅和硬件强相关，也和计算的形式强相关，因此写调度的时候还是要针对特定计算和特定后端硬件来写调度.
   * DSA核心困难是什么，是DSA本身的不稳定性和碎片化，这种编译算法一样会面临碎片化的问题.要想基于编译器设计一个针对DSA的可持续商业化路径，同时减少工程复杂性，我们还是要透过这些纷繁的具体技术选型去看整个栈从上到下有哪些问题，以及整个编译器领域总体的方法论以及能力边界
   * 编译器的本职工作是编译！编译器首先要解决的问题是如何将硬件无关的用户代码自动转换成硬件指令。而优化则是在编译的问题解决得足够好的情况下锦上添花的功能。编译器在整个软件栈中占据一席的主要原因必然是因为其本职工作——解耦合！
   * 好的软件栈是要让各个领域的人划分好边界，在自己关注的优化空间内自由发挥。算子编译器和算子库一定程度上就是一种竞争性的关系，算子编译器做得越强，覆盖的长尾算子就越多，算子库做得越全面，未覆盖的算子就越少，因为两者竞争的是同一块优化空间，结果就是算子编译器团队也会写很多手动优化塞进编译器里，算子库团队也会尝试自动化很多优化策略来扩充算子库。实际上这一块优化空间，编译器抢起来很难，算子库覆盖起来也很累。本该解耦合共同协作的各方很容易变成相互抢活的内卷。
   * DSA指令不仅可以包括芯片上实际的物理指令，也可以是高性能库提供的算子，硬件指令和手写库可以共同构建一套层次化的后端模板集合
   * 例子，如果我们现在处于算法专家还没提出某个算子的年代，我们软件栈怎么确保算法专家哪天拍脑袋想出这个算子的时候可以写出能这个算子，在当前硬件上用得还可以，发了论文宣传了这个算子的效果。接着各种大大小小的前端框架和应用开始重新实现这个算子以及这个算子的各个变种，下一代硬件决定增加专门的指令提升这个算子的性能。我们需要做到的是让这些大大小小的框架实现的各种该算子的变种都能自动利用起新指令来实现硬件加速，而且不能硬件等编译器，也不能应用开发等编译器。更重要的是，这里面每一层都不是一个公司、一个团队和一个框架，而是全行业大大小小的公司、团队和开发者    

6. Ray Tracing 08年的时候就是个热门话题，直到十几年后NV才做成产品. 最开始客户应该都用 cuda 自己写算法， 2080ti 开始有了硬件 RT core。 生态推动。    
7. tensor core wmma 编程也会转成ptx, 再编译优化调用 tensor core    
8. Intel 第四代至强 cpu 已经支持 CXL 1.0， 有的支持 2.0，  最新的 dGPU 也支持 CXL ，   CPU 也能高速访问dGPU 显存是比 nvlink 优势的地方。 pcie 物理层，增加了 io, cache, memory 
9. 关于 为什么需要 layout 转换    
    Layout 的选择 和 指令集设计 <-register blocking size<- caching blocking size 相关，比如一个指令做 16*4 FMA ( instruction 对应的应该是 register blocking， 毕竟指令主要操作的是 registers)     
    选择不同 layout 的目标是    
   * a. Register blocking：  提高 FMA指令/load-store 的比例， 并且能让 FMA 指令隐藏在 load/store 指令中
   * b. Caching blocking： 根据 缓存大小限制，争取 memory reuse 内存重用，找到 最小 Bytes/Flops (floating-point operations) 比
    Tensor Core, tensorRT,  华为的 达芬奇 core， 都有大量的 layout 转换， 也有 format 转换 （低精度性能不一定比高精度的性能好），  https://oneapi-src.github.io/oneDNN/enum_dnnl_format_tag_t.html   这个是 intel 的 layout， 好多好多，估计把 cpu avx 512， vnni， amx， gpu 的 EU， XMX 都包括进去了     
    a 是计算核心的硬件设计考虑的，b 是软硬件联合设计的问题，是performance model里面非常关键的部分
10. wmma： https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu    
    Tensor core 一个时钟周期4x4x4=64 FMA 乘加计算（相当于两个 4x4 矩阵乘）并累加到另一个 4x4 tensor 上 （D = A * B + C） ，每个 SM 8 个 tensor core， 就是 8x64/clock    
    Volta 架构会算 8x4 和 4x8 两个矩阵乘和累加， 即 m8n8k4， 就需要 4 个时钟周期    
    Cuda runtime 为了使指令overlap 更高，提升并行效率，把 m8n8k4 提升为 m16n16k16， 所以 wmma:mma_sync 是以 16x16 为最小单元    
    Wmma 接口以 warp 为单位每 4 个时钟周期向 tensor core 提交 m8n8k4 矩阵乘加  
    intel 的  XMX programing model 提供了 Joint_matrix API，   类似于 WMMA    
    Joint_matrix API -> LLVM IR, SPIRV -> GPU intrinsics -> GPUassembly -> GPU XMX    
    Wmma -> LLVM IR, SPIRV -> PTX intrinsics -> GPU assembly-> Tensor Core    
    sample： https://github.com/intel/llvm-test-suite/blob/intel/SYCL/Matrix/joint_matrix_bf16.cpp    
    下面有 LevelZero （driver支持SPIR-V）, 类似于 cuda runtime. OpenCL 也属于同一个 level，另外一条线， 我理解     
    关于指令 可以参考  AMX,  cpu 这边的， 硬件实现和 gpu/xmx 一样的， 只是使用上有些限制 https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=7258&techs=AMX    
    另外还提供了  dpc++ ESIMD API,  即 explicit SIMD SYCL Extention， 可以更贴近操控硬件。    
    真正形成决定性优势的，是对需求场景的足够理解打磨之后，在硬件上针对性的调整某些关键部分，在软件上加强生态优势    
11. XMX（Tensor core） 有两种编程模式， 一种是 用 joint_matrix API， 另一种是 ESIMD
    a. joint_matrix API （类似于 wmma），除了 XMX， 也可以用于 AMX， tensor core    
       https://github.com/intel/llvm-test-suite/blob/intel/SYCL/Matrix/    -> joint_matrix_tensorcore.cpp    
      RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3 %s -o %t.out     
      so for nvidia tensorcores, the implementation is AOT (not JIT), so you have to specify the target at compile time.    
      joint_matrix API 用的是  IGC compiler   （Intel graphics compiler）

    b. ESIMD    
   https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/optimization-and-programming/vectorization/explicit-vector-programming/explicit-simd-sycl-extension.html    
   Sample code:  https://github.com/intel/llvm-test-suite/blob/intel/SYCL/ESIMD/dpas/dpas_test3.cpp   (XMX 之前叫 dpas， Dot Product Accumulate Systolic)    
   ESIMD 用的是 CM compiler,  CM runtime -> Gen common ISA  （还没搞清楚是什么）    
   这个我理解主要是为生态做的， 当然也能用于 tensor core， 没找到 sample    目前来看， Joint_matrix API 的性能比 ESIMD 的性能要好
   数据流往往是制约性能的关键    计算部分都是编译器能解决的    

   1. Intel 最新的 GPU 把 类似于 nvidia 的 share memory 去掉了，只剩下 L1 cache， 和 大量的寄存器，    
   2. 所以 joint_matrix API 编程 tile 的 size 是有限制的， M 最大是8， N 最大是 16， K 是 16？， 直接从 Register File 分配 memory    
   3. tiling size (padding size) 可以通过 query inferface 选取最佳的 MKN， 也可以用 default 的     
   4. Joint_matrix API 也提供了灵活的指定，比如 layout， pre-pack （int8,fp16 pack 到 32 bytes）， size， stride 等指定这些信息， 也可以扩展 custom layout，tile 等      
   5. Nvidia 的 warp size 是固定的 32， intel gpu 的 subgroup 可以是 anysize， 有 joint 的概念，定义多少个 work items 为一个 subgroup， sycl 有更丰富的语义    
   6. Matrix 矩阵乘 结尾的 elementwise 操作也支持了， 类似于 nvidia 的 epilogue， 比如矩阵乘的结果乘以一个标量 alpha， 也是向量操作， 在 vector 部件上执行     
   7. Joint_matrix API 也支持 nvidia 的 tensor core， joint_matrix API 会直接调用 wmma API.         Joint_matrix 本身会由 compiler 转成 SPIRV matrix， 再有第二个 compiler 生成 binary     

12. LLVM IR， SPIR-V，还有 NV 的 NVVM IR，https://llvm.org/docs/tutorial    
    这个教程写得非常好，我把 Kaleidoscope， 实现的一种 language 的 8 个 examples 学习跑了一遍， 包括 词法分析，paser， AST， 生成 LLVM IR，JIT 实现，增加 optimizaiton pass 比如常量折叠，公共子表达式消除 CSE 等， 增加控制语句，自定义 operator,  还能实现交叉编译！ 很有意思~~ 

