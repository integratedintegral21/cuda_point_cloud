#include "PointCloud.hpp"

#include <numeric>
#include <stdexcept>
#include "iostream"

namespace cuda_point_cloud {

template<size_t I, size_t N, typename... ScalarTs>
void fill_buff(char *buf, const std::tuple<ScalarTs...> &scalar) {
  if constexpr (I == N) {
    return;
  } else {
    auto elem = std::get<I>(scalar);
    auto casted_buf = (decltype(elem)*)(buf);
    *casted_buf = elem;
    fill_buff<I + 1, N, ScalarTs...>((char*)(casted_buf + 1), scalar);
  }
}

template<typename... ScalarTs>
CudaPointCloud<ScalarTs...>::CudaPointCloud(
    const std::vector<PointCoord> &point_data) requires (!HAS_SCALARS_) {
  pcl_size_ = point_data.size();
  if (!point_data.empty()) {
    InitPoints(point_data);
  }
  for (size_t size: scalar_sizes_) {
    std::cout << size << std::endl;
  }
}

template<typename... ScalarTs>
CudaPointCloud<ScalarTs...>::CudaPointCloud(
    const std::vector<PointCoord> &point_data,
    const std::vector<ScalarsT> &scalar_data) requires HAS_SCALARS_ {
  if (point_data.size() != scalar_data.size()) {
    throw std::invalid_argument("Point data length different from scalar data length");
  }

  pcl_size_ = point_data.size();
  scalar_sizes_ = {sizeof(ScalarTs)...};
  if (!point_data.empty()) {
    InitPoints(point_data);
    InitScalars(scalar_data);
  }
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
void CudaPointCloud<ScalarTs...>::InitScalars(const std::vector<ScalarsT> &scalar_data)
requires HAS_SCALARS_{
  size_t pcl_size = scalar_data.size();
  size_t mem_size = pcl_size * sizeof(ScalarsT);
  cudaThrowIfStatusNotOk(cudaMalloc(&scalar_ptr_, mem_size));
  char *host_buf = (char*)malloc(mem_size);
  if (host_buf == nullptr) {
    throw std::runtime_error("Could not allocate host memory");
  }
  size_t stride = std::reduce(scalar_sizes_.begin(), scalar_sizes_.end());
  for (int i = 0; i < pcl_size; i++) {
    char *buf_start = host_buf + i * stride;
    auto scalar = scalar_data[i];
    fill_buff<0, sizeof...(ScalarTs), ScalarTs...>(buf_start, scalar);
  }

  cudaThrowIfStatusNotOk(cudaMemcpy(scalar_ptr_, host_buf, mem_size, cudaMemcpyHostToDevice));

  free(host_buf);
}

template<typename... ScalarTs>
void CudaPointCloud<ScalarTs...>::cudaThrowIfStatusNotOk(cudaError_t e) {
  if (e) {
    throw std::runtime_error(cudaGetErrorString(e));
  }
}

}  // namespace cuda_point_cloud
