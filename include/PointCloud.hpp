#ifndef CUDA_POINT_CLOUD_POINTCLOUD_HPP
#define CUDA_POINT_CLOUD_POINTCLOUD_HPP

#include <vector>
#include <cstdint>
#include <tuple>

#include <cuda_runtime_api.h>

#define INSTANTIATE_CUDA_POINT_CLOUD(...) \
    template class CudaPointCloud<__VA_ARGS__>;

namespace cuda_point_cloud {

struct PointCoord {
  float x, y, z;
};

template <typename ...ScalarTs>
class CudaPointCloud {
 private:
  typedef std::tuple<ScalarTs...> ScalarsT;
  constexpr static bool HAS_SCALARS_ = sizeof...(ScalarTs) > 0;

 public:
  explicit CudaPointCloud(const std::vector<PointCoord> &point_data) requires (!HAS_SCALARS_);
  CudaPointCloud(const std::vector<PointCoord> &point_data,
                 const std::vector<ScalarsT> &scalar_data) requires HAS_SCALARS_;

  ~CudaPointCloud();

 private:
  void *xyz_ptr_ = nullptr;
  void *scalar_ptr_ = nullptr;

  // Helpers
  void InitPoints(const std::vector<PointCoord> &point_data);
  void InitScalars(const std::vector<ScalarsT> &scalar_data);
  void cudaThrowIfStatusNotOk(cudaError_t e);
};

typedef CudaPointCloud<> CudaPointCloudXYZ;
INSTANTIATE_CUDA_POINT_CLOUD()

typedef CudaPointCloud<float> CudaPointCloudXYZI;
INSTANTIATE_CUDA_POINT_CLOUD(float)

typedef CudaPointCloud<uint8_t, uint8_t, uint8_t> CudaPointCloudXYZRGB;
INSTANTIATE_CUDA_POINT_CLOUD(uint8_t, uint8_t, uint8_t)

typedef CudaPointCloud<float, float, float> CudaPointCloudXYZRGBFloat;
INSTANTIATE_CUDA_POINT_CLOUD(float, float, float)

}  // namespace cuda_point_cloud

#endif  // CUDA_POINT_CLOUD_POINTCLOUD_HPP
