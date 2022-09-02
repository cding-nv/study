
# 安装pytorch cuda extension
1. cd OODNLib
2. python setup.py install  
如果要调试extension代码，要确保setup.py中的debug为True


# 在PAI-Pytorch容器中编译whl包
1. 添加docker用户组  
sudo gpasswd -a \`whoami\` docker
2. 下载PAI-Pytorch镜像，二选一  
docker pull reg.docker.alibaba-inc.com/pai-pytorch/pytorch-release:1.5.1-20200721.gpu_cuda10.1  
docker pull reg.docker.alibaba-inc.com/pai-pytorch/pytorch-release:1.8.0-202107071840_cuda10.1
3. 以交互方式启动docker容器，并将宿主机上的OODNLib所在目录挂载到容器的workspace，二选一  
nvidia-docker run -it --rm -v \`pwd\`:/workspace --shm-size 16G --net=host reg.docker.alibaba-inc.com/pai-pytorch/pytorch-release:1.5.1-20200721.gpu_cuda10.1 bash  
nvidia-docker run -it --rm -v \`pwd\`:/workspace --shm-size 16G --net=host reg.docker.alibaba-inc.com/pai-pytorch/pytorch-release:1.8.0-202103311800_cuda10.1 bash  
4. 激活python环境  
source /opt/conda/bin/activate python3.6
5. 进入workspace  
cd /workspace/OODNLib
6. 编译whl包  
python setup.py bdist_wheel
7. 复制whl包  
cp ./dist/*.whl ..
8. 如果在宿主机上没有root权限，需要通过chown修改whl包的所有者，再exit退出容器


# 同步测试代码和结果
1. cd Test/FunctionTest
2. git submodule update
3. git checkout master


# 调试pytorch cuda extension
以PrecisionTest为例
1. 以debug模式安装extension
2. cd Test/FunctionTest/PrecisionTest
3. 在main.py中添加断点函数  
def breakpoint():  
import os, signal  
os.kill(os.getpid(), signal.SIGTRAP)  
4. 在调用extension之前调用断点函数
5. 启动调试环境  
   gdb python
6. 执行到刚才插入的python断点  
   run main.py tensor_fp32_bsz020
7. 插入C++断点, MultimodalAttention_init是C++函数名  
   b MultimodalAttention_init
8. 继续执行到C++断点  
   c   
9. 此时就已经进入到cuda extension代码了，就和调试C++一样了。  
这种方式不能进入cuda核函数内部。要调试CUDA代码，第5步要改为cuda-gdb python，并且要求CUDA版本为10.2及以上。 
 
