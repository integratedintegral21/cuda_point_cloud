#ifndef CUDA_POINT_CLOUD_FILTER_HPP
#define CUDA_POINT_CLOUD_FILTER_HPP

#include <functional>
#include <tuple>

#include "point_cloud.hpp"

#define INSTANTIATE_FILTER(...) \
  template void filter_by_coordinates<__VA_ARGS__>( \
    const CudaPointCloud<__VA_ARGS__> &in_pcl, \
    CudaPointCloud<__VA_ARGS__> &out_pcl, \
    std::function<bool(const PointCoord&)> &&filter_fun);

namespace cuda_point_cloud {

template<typename ...ScalarTs>
void filter_by_coordinates(const CudaPointCloud<ScalarTs...> &in_pcl,
                           CudaPointCloud<ScalarTs...> &out_pcl,
                           std::function<bool(const PointCoord&)> &&filter_fun);

template<typename ...ScalarTs>
void filter_by_scalars(const CudaPointCloud<ScalarTs...> &in_pcl,
                       CudaPointCloud<ScalarTs...> &out_pcl,
                       std::function<bool(const std::tuple<ScalarTs...> &)> &&filter_fun);

template<typename ...ScalarTs>
void filter(const CudaPointCloud<ScalarTs...> &in_pcl,
            CudaPointCloud<ScalarTs...> &out_pcl,
            std::function<bool(const PointCoord&, const std::tuple<ScalarTs...> &)> &&filter_fun);

}  // namespace cuda_point_cloud

#endif  // CUDA_POINT_CLOUD_FILTER_HPP
