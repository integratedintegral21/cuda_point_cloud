cmake_minimum_required(VERSION 3.29)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cuda_point_cloud CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

add_library(cuda_point_cloud SHARED
        src/PointCloud.cpp
)
target_include_directories(cuda_point_cloud PUBLIC include)
set_target_properties(cuda_point_cloud PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
