#include "transform.hpp"

#include <cuda_runtime_api.h>

namespace cuda_point_cloud {

template <typename ...ScalarTs>
void transform_point_cloud(const CudaPointCloud<ScalarTs...> &in_pcl,
                           CudaPointCloud<ScalarTs...> &out_pcl,
                           Eigen::Matrix<float, 3, 4, Eigen::ColMajor> &transform) {
  if (in_pcl.Size() == 0) {
    return;
  }


}


INSTANTIATE_CUDA_POINT_CLOUD();
INSTANTIATE_CUDA_POINT_CLOUD(float);
INSTANTIATE_CUDA_POINT_CLOUD(uint8_t, uint8_t, uint8_t);
INSTANTIATE_CUDA_POINT_CLOUD(float, float, float);

}  // namespace cuda_point_cloud
