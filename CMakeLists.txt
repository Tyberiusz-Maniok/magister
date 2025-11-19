cmake_minimum_required(VERSION 3.20)
enable_testing()
project(magister_ex LANGUAGES CXX)

set(CMAKE_C_COMPILER "/opt/llvm_rl/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/llvm_rl/bin/clang++")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IMAGEMAGICK REQUIRED ImageMagick++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -gdwarf-4 --libomptarget-nvptx-bc-path=/opt/llvm_rl/lib/nvptx64-nvidia-cuda/libomptarget-nvptx.bc -fuse-ld=/opt/llvm_rl/bin/ld.lld -D__noinline__= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(magister_ex "${SRC}")


target_compile_options(magister_ex PUBLIC 
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
)

target_include_directories(magister_ex PUBLIC 
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    ${IMAGEMAGICK_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(magister_ex PUBLIC
    $<LINK_ONLY:MKL::MKL>
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
    ${IMAGEMAGICK_LIBRARIES}
)

target_link_options(magister_ex PRIVATE 
    "-Wl,-rpath,/opt/llvm_rl/lib"
    "-Wl,-rpath,/opt/llvm_rl/lib/x86_64-unknown-linux-gnu"
)

set_target_properties(magister_ex PROPERTIES
    BUILD_RPATH "/opt/llvm_rl/lib/x86_64-unknown-linux-gnu"
    INSTALL_RPATH "/opt/llvm_rl/lib/x86_64-unknown-linux-gnu"
)

if(OpenMP_CXX_FOUND)
    target_compile_options(magister_ex PRIVATE
        -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        -Xopenmp-target -march=sm_80
    )
endif()
