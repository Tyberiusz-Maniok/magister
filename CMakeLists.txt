cmake_minimum_required(VERSION 3.13)
enable_testing()
project(magister_ex)
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP)
#message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(magister_ex "${SRC}")

target_compile_options(magister_ex PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(magister_ex PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(magister_ex PUBLIC $<LINK_ONLY:MKL::MKL> $<$<CONFIG:Release>:OpenMP::OpenMP_CXX>)

target_include_directories(magister_ex PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_test(NAME mytest COMMAND myapp)
if(MKL_ENV)
  set_tests_properties(mytest PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif()

if(OpenMP_CXX_FOUND)
  # target_link_libraries(magister_ex $<$<CONFIG:Release>:OpenMP::OpenMP_CXX>)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
