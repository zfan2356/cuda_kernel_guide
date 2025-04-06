cmake_minimum_required(VERSION 3.26.5)

set(CMAKE_CUDA_COMPILER /data/home/xingonzhang/micromamba/envs/dev1/bin/nvcc)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/data/home/xingonzhang/micromamba/envs/dev1/targets/x86_64-linux/include)
link_directories(/data/home/xingonzhang/micromamba/envs/dev1/targets/x86_64-linux/lib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cuda_test LANGUAGES CXX CUDA)

add_executable(hello_world hello_world.cu)
add_executable(vector_add vector_add.cu)
add_executable(reduce reduce.cu)
add_executable(gelu gelu.cu)
