#ifndef CUDA_POINT_CLOUD_POINTCLOUD_HPP
#define CUDA_POINT_CLOUD_POINTCLOUD_HPP

#include <vector>

#include <cuda.h>

namespace cuda_point_cloud {

template <typename ...ScalarTs>
class CudaPointCloud {
 public:
  CudaPointCloud(const std::vector<float> &point_data,);
  ~CudaPointCloud() {
    if (xyz_ptr) {

    }
  }

 private:
  void *xyz_ptr = nullptr;
  void *scalar_ptr = nullptr;
};

}  // namespace cuda_point_cloud

#endif  // CUDA_POINT_CLOUD_POINTCLOUD_HPP
