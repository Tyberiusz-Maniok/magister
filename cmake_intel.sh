#!/bin/sh
conda activate icpx
cd ./build
source /opt/intel/oneapi/setvars.sh
# cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..
cmake -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib -DCMAKE_CUDA_COMPILER=nvcc ..

# source /opt/intel/oneapi/vtune/latest/env/vars.sh
# sudo sysctl -w kernel.yama.ptrace_scope=0

# export MKL_VERBOSE=1
# export LIBOMPTARGET_INFO=4
# export MKL_ENABLE_CBLAS_OFFLOAD=1

make
