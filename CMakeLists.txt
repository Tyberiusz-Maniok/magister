cmake_minimum_required(VERSION 3.20)

# CRITICAL: Set compilers BEFORE project() call
set(CMAKE_C_COMPILER "/opt/llvm18/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/llvm18/bin/clang++")
set(CMAKE_CXX_STANDARD 17)

enable_testing()
project(magister_ex LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IMAGEMAGICK REQUIRED ImageMagick++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -gdwarf-4 -fuse-ld=/opt/llvm18/bin/ld.lld")

# Workaround: Define __noinline__ as empty to prevent CUDA's incompatible definition
add_compile_definitions(__noinline__=)

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(magister_ex "${SRC}")


target_include_directories(magister_ex PUBLIC 
    ${IMAGEMAGICK_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(magister_ex PUBLIC
    CUDA::cublas
    CUDA::cudart
    ${IMAGEMAGICK_LIBRARIES}
)

target_link_options(magister_ex PRIVATE 
    "-Wl,-rpath,/opt/llvm18/lib"
    "-L/opt/llvm18/lib"
    "/opt/llvm18/lib/libomp.so"
    "/opt/llvm18/lib/libomptarget.so"
    "-Wl,--as-needed"
)

set_target_properties(magister_ex PROPERTIES
    BUILD_RPATH "/opt/llvm18/lib"
    INSTALL_RPATH "/opt/llvm18/lib"
)

if(OpenMP_CXX_FOUND)
    target_compile_options(magister_ex PRIVATE
        -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        -Xopenmp-target -march=sm_86
    )
endif()
