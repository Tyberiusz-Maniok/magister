#!/bin/sh
conda activate icpx
cd ./build
source /opt/intel/oneapi/setvars.sh
# cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..
# cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib -DCMAKE_CUDA_COMPILER=nvcc ..
cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib ..
# cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib -DOpenCV_DIR=$HOME/Projects/opencv_install/lib/cmake/opencv4 -DCMAKE_CUDA_COMPILER=nvcc ..

# CC=/home/tyberiuszm/Projects/llvm-project/build2/bin/clang CXX=/home/tyberiuszm/Projects/llvm-project/build2/bin/clang++ cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib ..

# source /opt/intel/oneapi/vtune/latest/env/vars.sh
# sudo sysctl -w kernel.yama.ptrace_scope=0

# export MKL_VERBOSE=1
# export LIBOMPTARGET_INFO=4
# export MKL_ENABLE_CBLAS_OFFLOAD=1

make


# build clang
# cmake ../llvm -DCMAKE_BUILD_TYPE=Release -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_80 -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES="80,86" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="openmp;offload;libc"

