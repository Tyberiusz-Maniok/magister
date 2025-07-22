#!/bin/sh
conda activate icpx
cd ./build
rm -rf *
source /opt/intel/oneapi/mkl/latest/env/vars.sh
source /opt/intel/oneapi/compiler/latest/env/vars.sh
# source /opt/intel/oneapi/vtune/latest/env/vars.sh
make
