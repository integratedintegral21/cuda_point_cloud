cmake_minimum_required(VERSION 3.27)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cuda_point_cloud CUDA CXX)

enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_FLAGS --extended-lambda)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${EIGEN3_INCLUDE_DIRS})

add_library(cuda_point_cloud SHARED
        src/point_cloud.cpp
        src/transform.cu
        src/filter.cu
)

target_include_directories(cuda_point_cloud PUBLIC
        include)
target_link_libraries(cuda_point_cloud PUBLIC
        Eigen3::Eigen)
set_target_properties(cuda_point_cloud PROPERTIES CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON)

# Unit tests
add_executable(cuda_point_cloud_test
        test/cuda_point_cloud_test.cpp)
target_include_directories(cuda_point_cloud_test PUBLIC include)
target_link_libraries(cuda_point_cloud_test
        cuda_point_cloud
        GTest::gtest_main
        /usr/local/cuda/lib64/libcudart.so
        Eigen3::Eigen)

add_executable(filter_test
        test/filter_test.cpp)
target_include_directories(filter_test PUBLIC include)
target_link_libraries(filter_test
        cuda_point_cloud
        GTest::gtest_main
        /usr/local/cuda/lib64/libcudart.so
        Eigen3::Eigen)

include(GoogleTest)
gtest_add_tests(TARGET cuda_point_cloud_test)
gtest_add_tests(TARGET filter_test)
