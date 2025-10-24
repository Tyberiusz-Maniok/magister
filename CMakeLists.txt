cmake_minimum_required(VERSION 3.20)
enable_testing()
project(magister_ex LANGUAGES CXX)

# set(CMAKE_C_COMPILER "/usr/local/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/local/bin/clang++")
set(CMAKE_C_COMPILER "/opt/llvm_a/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/llvm_a/bin/clang++")
# set(CMAKE_LINKER "/opt/llvm_a/bin/lld")
# set(CMAKE_LINKER_TYPE LLD)



find_package(CUDAToolkit REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(IntelDPCPP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IMAGEMAGICK REQUIRED ImageMagick++)

#message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math --libomptarget-nvptx-bc-path=/home/tyberiuszm/Projects/llvm-project/build_2/lib/libomptarget-nvptx.bc")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math --libomptarget-nvptx-bc-path=/opt/llvm_a/lib/nvptx64-nvidia-cuda/libomptarget-nvptx.bc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g --libomptarget-nvptx-bc-path=/opt/llvm_a/lib/nvptx64-nvidia-cuda/libomptarget-nvptx.bc -stdlib=libc++ -I/usr/include/c++/v1 -fuse-ld=/opt/llvm_a/bin/ld.lld")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -fcf-protection=none")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -fcf-protection=none -foffload=nvptx-none -march=sm_80")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -fsanitize=address")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC -lineinfo")

# set(CMAKE_CUDA_ARCHITECTURES 80)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(magister_ex "${SRC}")

# set_target_properties(magister_ex PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
#   POSITION_INDEPENDENT_CODE ON
# )

target_compile_options(magister_ex PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(magister_ex PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES> ${IMAGEMAGICK_INCLUDE_DIRS})
# target_link_libraries(magister_ex PUBLIC $<LINK_ONLY:MKL::MKL> CUDA::cublas CUDA::cudart $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> ${OpenCV_LIBS})

find_library(OMPTARGET_LIB
  NAMES omptarget libomptarget
  PATHS /opt/llvm_a/lib /usr/local/lib /usr/lib
  NO_DEFAULT_PATH
)

if(OMPTARGET_LIB)
  message(STATUS "Found libomptarget: ${OMPTARGET_LIB}")
  target_link_libraries(magister_ex PRIVATE ${OMPTARGET_LIB})
  # Ensure the runtime path is set so loader can find the lib at run-time
  target_link_options(magister_ex PRIVATE "-Wl,-rpath,/opt/llvm_a/lib")
else()
  message(WARNING "libomptarget not found. Link failures for OpenMP offload symbols are likely.")
endif()

target_link_libraries(magister_ex PUBLIC $<LINK_ONLY:MKL::MKL> CUDA::cublas CUDA::cudart $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> ${IMAGEMAGICK_LIBRARIES} -L/opt/llvm_a/lib -lomptarget)
target_include_directories(magister_ex PUBLIC
    ${IMAGEMAGICK_INCLUDE_DIRS}
)

target_link_libraries(magister_ex PUBLIC
    $<LINK_ONLY:MKL::MKL>
    CUDA::cublas CUDA::cudart
    OpenMP::OpenMP_CXX
    ${IMAGEMAGICK_LIBRARIES}
)
target_link_options(magister_ex PRIVATE "-Wl,-rpath,/opt/llvm_a/lib")

# target_compile_options(magister_ex PRIVATE -Werror -Wall -Wextra)

include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(magister_ex PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_test(NAME mytest COMMAND myapp)
if(MKL_ENV)
  set_tests_properties(mytest PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif()

if(OpenMP_CXX_FOUND)
  # target_link_libraries(magister_ex $<$<CONFIG:Release>:OpenMP::OpenMP_CXX>)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -lc++abi")
endif()

if(OpenMP_CXX_FOUND)
    target_compile_options(magister_ex PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -fopenmp
            -fopenmp-targets=nvptx64-nvidia-cuda
            -Xopenmp-target -march=sm_80
        >
    )
    target_link_libraries(magister_ex PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(magister_ex PROPERTIES
    BUILD_RPATH "/opt/llvm_a/lib/x86_64-unknown-linux-gnu"
    INSTALL_RPATH "/opt/llvm_/lib/x86_64-unknown-linux-gnu"
)
