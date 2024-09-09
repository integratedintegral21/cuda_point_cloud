#ifndef CUDA_POINT_CLOUD_POINTCLOUD_HPP
#define CUDA_POINT_CLOUD_POINTCLOUD_HPP

#include <vector>
#include <tuple>

#include <cuda.h>

namespace cuda_point_cloud {

template <typename ...ScalarTs>
class CudaPointCloud {
 private:
  constexpr static bool HAS_SCALARS = sizeof...(ScalarTs);

 public:
  explicit CudaPointCloud(const std::vector<float> &point_data) requires (!HAS_SCALARS);
  CudaPointCloud(const std::vector<float> &point_data,
                 const std::vector<std::tuple<ScalarTs...>> &scalar_data) requires HAS_SCALARS;

  ~CudaPointCloud() {
    if (xyz_ptr) {
      cudaFree(xyz_ptr);
    }
    if (scalar_ptr) {
      cudaFree(scalar_ptr);
    }
  }

 private:
  void *xyz_ptr = nullptr;
  void *scalar_ptr = nullptr;
};

}  // namespace cuda_point_cloud

#endif  // CUDA_POINT_CLOUD_POINTCLOUD_HPP
