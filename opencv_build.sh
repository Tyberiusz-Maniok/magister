#!/bin/bash
# Build OpenCV with CUDA support for sm_80

# Adjust these paths as needed
INSTALL_DIR=$HOME/Projects/opencv_install
SRC_DIR=$HOME/Projects/opencv_source

# Get source (if you don’t have it yet)
# git clone https://github.com/opencv/opencv.git $SRC_DIR
# git clone https://github.com/opencv/opencv_contrib.git $SRC_DIR/opencv_contrib

mkdir -p $SRC_DIR/build
cd $SRC_DIR/build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -D OPENCV_EXTRA_MODULES_PATH=$SRC_DIR/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_ARCH_BIN=8.0 \
      -D CUDA_ARCH_PTX=8.0 \
      -D BUILD_opencv_cudacodec=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D CMAKE_C_COMPILER=/usr/bin/gcc-11 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-11 \
      ..

make -j$(nproc)
make install
