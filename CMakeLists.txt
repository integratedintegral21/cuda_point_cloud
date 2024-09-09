cmake_minimum_required(VERSION 3.29)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cuda_point_cloud CUDA CXX)

enable_testing()
set(CMAKE_CXX_STANDARD 20)

find_package(GTest REQUIRED)

add_library(cuda_point_cloud SHARED
        src/PointCloud.cpp
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(cuda_point_cloud PUBLIC
        include)
set_target_properties(cuda_point_cloud PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Unit tests
add_executable(cuda_point_cloud_test
        test/cuda_point_cloud_test.cpp)
target_include_directories(cuda_point_cloud_test PUBLIC include)
target_link_libraries(cuda_point_cloud_test GTest::gtest_main cuda_point_cloud)

include(GoogleTest)
gtest_add_tests(TARGET cuda_point_cloud_test)
