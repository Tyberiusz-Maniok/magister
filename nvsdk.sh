#!/bin/bash

# NVIDIA HPC SDK 25.1
export NVARCH=Linux_x86_64
export NVCOMPILERS=/opt/nvidia/hpc_sdk  
export NVHPC_VERSION=25.1  # Adjust to actual installed version

export PATH=$NVCOMPILERS/$NVARCH/$NVHPC_VERSION/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVCOMPILERS/$NVARCH/$NVHPC_VERSION/compilers/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVCOMPILERS/$NVARCH/$NVHPC_VERSION/cuda/12.6/lib64:$LD_LIBRARY_PATH

# For OpenMP
export OMP_DEFAULT_DEVICE=0
