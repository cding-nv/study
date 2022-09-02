import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

DEBUG = False
BUILD_ALL = True


def cc_list(build_all=BUILD_ALL):
    if build_all:
        ccs = [60, 61, 70]
        cuda_version = float(torch.version.cuda)
        if cuda_version >= 10:
            ccs.append(75)
        if cuda_version >= 11:
            ccs.append(80)
        if cuda_version >= 11.1:
            ccs.append(86)
    else:
        ccs = []
        cnt = torch.cuda.device_count()
        for i in range(cnt):
            sm = torch.cuda.get_device_capability(i)
            cc = int(f"{sm[0]}{sm[1]}")
            if cc not in ccs and cc >= 60:
                ccs.append(cc)
    return ccs


def cxx_flags():
    if DEBUG:
        flags = ["-D_DEBUG", "-g"]
    else:
        flags = ["-O3"]
    flags += ["-std=c++14"]
    return flags


def nvcc_flags():
    if DEBUG:
        flags = ["-D_DEBUG", "-g", "-G"]
    else:
        flags = ["-O3"]
    flags += [
        "-std=c++14",
        "-use_fast_math",
        "-expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    for cc in cc_list():
        flags.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
    return flags


def gen_module(name, sources):
    ext = CUDAExtension(
        name=name,
        sources=sources,
        include_dirs=["csrc/include"],
        extra_compile_args={
            "cxx": cxx_flags(),
            "nvcc": nvcc_flags(),
        },
    )
    return ext


ext_modules = [
    gen_module(
        name="oodnlib.ext.bert",
        sources=[
            "csrc/Bert.cu",
            "csrc/CustomBert.cpp",
            "csrc/kernel/Add.cu",
            "csrc/kernel/BertEncoderInfer.cu",
            "csrc/kernel/BertPoolerInfer.cu",
            "csrc/kernel/DropPath.cu",
            "csrc/kernel/Dropout.cu",
            "csrc/kernel/Gelu.cu",
            "csrc/kernel/SliceSqueeze.cu",
        ],
    ),
    gen_module(
        name="oodnlib.ext.func",
        sources=[
            "csrc/Func.cu",
            "csrc/CustomFunc.cpp",
            "csrc/kernel/AdjMatrixBatch.cu",
            "csrc/kernel/BatchingSequence.cu",
            "csrc/kernel/PositionsAndTimeDiff.cu",
            "csrc/kernel/RecoverSequenceInfer.cu",
        ],
    ),
    gen_module(
        name="oodnlib.ext.mmsa",
        sources=[
            "csrc/MMSelfAttn.cu",
            "csrc/CustomMMSA.cpp",
            "csrc/kernel/Dropout.cu",
            "csrc/kernel/MMSelfAttnInferGL.cu",
            "csrc/kernel/MMSelfAttnInferL.cu",
            "csrc/kernel/UniOps.cu",
        ],
    ),
]


cmdclass = {
    "build_ext": BuildExtension.with_options(use_ninja=False, no_python_abi_suffix=True)
}


version = open("version.txt", "r").read().strip()
setup(
    name="oodnlib",
    version=version,
    description="OODN Library",
    author="xinghe",
    author_email="xinghe.yg@alibaba-inc.com",
    packages=find_packages(exclude=["csrc"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
