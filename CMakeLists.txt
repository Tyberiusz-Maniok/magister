cmake_minimum_required(VERSION 3.20)

# NVIDIA HPC SDK Paths
set(NVHPC_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9")
set(CMAKE_C_COMPILER "${NVHPC_ROOT}/compilers/bin/nvc")
set(CMAKE_CXX_COMPILER "${NVHPC_ROOT}/compilers/bin/nvc++")
set(CMAKE_CXX_STANDARD 17)

enable_testing()
project(magister_ex LANGUAGES CXX)

# NVIDIA HPC SDK compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fast -gopt")

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(magister_ex "${SRC}")

target_include_directories(magister_ex PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${NVHPC_ROOT}/cuda/12.6/include"
)

target_link_directories(magister_ex PUBLIC
    "${NVHPC_ROOT}/compilers/lib"
    "${NVHPC_ROOT}/math_libs/12.6/targets/x86_64-linux/lib"
    "${NVHPC_ROOT}/cuda/12.6/lib64"
)

target_link_libraries(magister_ex PUBLIC
    cublas
    cudart
)

set_target_properties(magister_ex PROPERTIES
    BUILD_RPATH "${NVHPC_ROOT}/compilers/lib:${NVHPC_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:${NVHPC_ROOT}/cuda/12.6/lib64"
    INSTALL_RPATH "${NVHPC_ROOT}/compilers/lib:${NVHPC_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:${NVHPC_ROOT}/cuda/12.6/lib64"
)

# OpenMP GPU offload flags (each flag as separate argument)
target_compile_options(magister_ex PRIVATE 
    -mp=gpu
    -gpu=cc86
    -Minfo=mp
)

# Link flags for GPU offload (must pass -mp=gpu to linker as well)
target_link_options(magister_ex PRIVATE
    -mp=gpu
    -gpu=cc86
)
