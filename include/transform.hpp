#ifndef CUDA_POINT_CLOUD_TRANSFORM_HPP
#define CUDA_POINT_CLOUD_TRANSFORM_HPP

#include <Eigen/Core>

#include "point_cloud.hpp"

#define INSTANTIATE_TRANSFORM(...) \
  template void transformPointCloud<__VA_ARGS__>(const CudaPointCloud<__VA_ARGS__> &, \
                                          CudaPointCloud<__VA_ARGS__> &,       \
                                          Eigen::Matrix<float, 3, 4, Eigen::ColMajor> &transform);

namespace cuda_point_cloud {
template <typename ...ScalarTs>
void transformPointCloud(const CudaPointCloud<ScalarTs...> &in_pcl,
                         CudaPointCloud<ScalarTs...> &out_pcl,
                         Eigen::Matrix<float, 3, 4, Eigen::ColMajor> &transform);

} // namespace cuda_point_cloud

#endif  // CUDA_POINT_CLOUD_TRANSFORM_HPP
