This package consists of the following files:
.
├── CMakeLists.txt
├── sample                  # the sample code
│   ├── test_custom.py      # call customized tf op to implement tf.einusm
│   └── test_orig.py        # Original tf einsum
└── src
    ├── cuda                # CUDA implementation
    │   ├── non_zero_index cuda kernel
    │   └── scatter_custom cuda kernel
    └── tf_op               # Tensorflow op implementation
        ├── 
        ├── non_zero_index, tf custom op to find which index is not zero line
        └── scatter_custom, tf custom op to replace tf.scatter_nd

How to build:
1. mkdir build
2. cd build
3. cmake ..
4. make
5. cd ./sample
6. python test_custom.py


Notes:
[20, 1, 512, 1024] x [20, 1024, 9984]
non_zero_index cuda kernel 20x512 个 thread, 每个 len = 1024, 只看第一第二个是否为 0.0f

custom scatter_nd 构建一个新的 tensor (初始化为0)，shape 自定义，将 value 按 index 写入， scatter_nd(indices, value, shape)

