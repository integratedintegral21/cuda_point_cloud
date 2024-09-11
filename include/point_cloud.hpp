#ifndef CUDA_POINT_CLOUD_POINT_CLOUD_HPP
#define CUDA_POINT_CLOUD_POINT_CLOUD_HPP

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
  std::vector<size_t> scalar_sizes_;

 public:
  CudaPointCloud();
  explicit CudaPointCloud(const std::vector<PointCoord> &point_data) requires (!HAS_SCALARS_);
  CudaPointCloud(const std::vector<PointCoord> &point_data,
                 const std::vector<ScalarsT> &scalar_data) requires HAS_SCALARS_;

  ~CudaPointCloud();

  // Getters
  [[nodiscard]] size_t Size() const {
    return pcl_size_;
  }

  [[nodiscard]] std::vector<size_t> ScalarSizes() const {
    return scalar_sizes_;
  }

  [[nodiscard]] PointCoord* PointCoordDevPtr() {
    return reinterpret_cast<PointCoord*>(xyz_ptr_);
  }

  [[nodiscard]] const PointCoord* PointCoordDevPtr() const {
    return reinterpret_cast<const PointCoord*>(xyz_ptr_);
  }

  [[nodiscard]] void* ScalarDevPtr() requires HAS_SCALARS_ {
    return scalar_ptr_;
  }

  [[nodiscard]] const void* ScalarDevPtr() const requires HAS_SCALARS_ {
    return scalar_ptr_;
  }

  void resize(size_t n);

  /**
   * A bit slow, use for debugging only
   * @return
   */
  [[nodiscard]] std::vector<PointCoord> GetHostPoints() const;

  /**
   * A bit slow, use for debugging only
   * @return
   */
  [[nodiscard]] std::vector<ScalarsT> GetHostScalars() const requires HAS_SCALARS_;

 private:
  size_t pcl_size_ = 0;
  PointCoord *xyz_ptr_ = nullptr;
  void *scalar_ptr_ = nullptr;

  // Helpers
  void InitPoints(const std::vector<PointCoord> &point_data);
  void InitScalars(const std::vector<ScalarsT> &scalar_data) requires HAS_SCALARS_;
  void cudaThrowIfStatusNotOk(cudaError_t e) const;
};

template<typename... ScalarTs>
CudaPointCloud<ScalarTs...>::CudaPointCloud() {
  scalar_sizes_ = {sizeof(ScalarTs)...};
}

typedef CudaPointCloud<> CudaPointCloudXYZ;
typedef CudaPointCloud<float> CudaPointCloudXYZI;
typedef CudaPointCloud<uint8_t, uint8_t, uint8_t> CudaPointCloudXYZRGB;
typedef CudaPointCloud<float, float, float> CudaPointCloudXYZRGBFloat;

}  // namespace cuda_point_cloud

#endif  // CUDA_POINT_CLOUD_POINT_CLOUD_HPP
