# Run
# $ sudo docker run -it -d  --privileged --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -e  https_proxy=$https_proxy -e no_proxy=$no_proxy  -v  /home/fengding/:/home/fengding --rm nvcr.io/nvidia/pytorch:22.12-py3
# $ python test_tp.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

TENSOR_PARALLEL_SIZE = 2  # 2个GPU进行张量并行

def tensor_parallel_linear(rank, world_size, W, x):
    """模拟 Tensor Parallel 线性层计算（使用已知整数矩阵）"""
    
    # 1. 初始化分布式环境
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 绑定 GPU

    # 2. 将 W 和 x 复制到当前 GPU
    W = W.to(f"cuda:{rank}")
    x = x.to(f"cuda:{rank}")

    # 3. 按列拆分 W，按行拆分 x
    W_shard = W.chunk(world_size, dim=1)[rank]  # 每个GPU负责一半列
    x_shard = x.chunk(world_size, dim=0)[rank]  # 每个GPU负责 x 的一半行

    # 4. 计算局部矩阵乘法
    partial_result = W_shard @ x_shard  # 局部计算

    # 5. 执行 all-reduce 操作，汇总最终结果
    dist.all_reduce(partial_result, op=dist.ReduceOp.SUM)

    # 6. Rank 0 打印最终结果
    if rank == 0:
        print(f"Final Output on Rank {rank}:\n", partial_result.cpu().numpy())

    # 7. 关闭分布式进程
    dist.destroy_process_group()

def main():
    world_size = TENSOR_PARALLEL_SIZE  # 需要的 GPU 数量

    # 定义固定整数权重矩阵 W
    W = torch.tensor([
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=torch.float32)  # 设定为 float32 以便计算

    # 定义输入向量 x
    x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # (4,1)

    # 启动多进程
    mp.spawn(tensor_parallel_linear, args=(world_size, W, x), nprocs=world_size)

if __name__ == "__main__":
    main()

