#include "PointCloud.hpp"

#include <stdexcept>

namespace cuda_point_cloud {

template<typename... ScalarTs>
CudaPointCloud<ScalarTs...>::CudaPointCloud(
    const std::vector<PointCoord> &point_data) requires (!HAS_SCALARS_) {
  InitPoints(point_data);
}

template<typename... ScalarTs>
CudaPointCloud<ScalarTs...>::CudaPointCloud(
    const std::vector<PointCoord> &point_data,
    const std::vector<ScalarsT> &scalar_data) requires HAS_SCALARS_ {
  if (point_data.size() != scalar_data.size()) {
    throw std::invalid_argument("Point data length different from scalar data length");
  }
  InitPoints(point_data);
}

template<typename... ScalarTs>
CudaPointCloud<ScalarTs...>::~CudaPointCloud() {
  if (xyz_ptr_) {
    cudaFree(xyz_ptr_);
  }
  if (scalar_ptr_) {
    cudaFree(scalar_ptr_);
  }
}

template<typename... ScalarTs>
void CudaPointCloud<ScalarTs...>::InitPoints(const std::vector<PointCoord> &point_data) {
  size_t pcl_size = point_data.size();
  size_t mem_size = pcl_size * sizeof(PointCoord);
  cudaThrowIfStatusNotOk(cudaMalloc(&xyz_ptr_, mem_size));

  auto xyz_host_ptr_ = point_data.data();
  cudaThrowIfStatusNotOk(cudaMemcpy(xyz_ptr_, xyz_host_ptr_, mem_size, cudaMemcpyHostToDevice));
}

template<typename... ScalarTs>
void CudaPointCloud<ScalarTs...>::InitScalars(const std::vector<ScalarsT> &scalar_data) {
  size_t pcl_size = scalar_data.size();
  size_t mem_size = pcl_size * sizeof(ScalarsT);
  cudaThrowIfStatusNotOk(cudaMalloc(&scalar_ptr_, mem_size));
}

template<typename... ScalarTs>
void CudaPointCloud<ScalarTs...>::cudaThrowIfStatusNotOk(cudaError_t e) {
  if (e) {
    throw std::runtime_error(cudaGetErrorString(e));
  }
}

}  // namespace cuda_point_cloud
