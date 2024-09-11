#include "point_cloud.hpp"

#include <numeric>
#include <stdexcept>
#include "iostream"

namespace cuda_point_cloud {

template<size_t I, typename... ScalarTs>
void fill_buff(char *buf, const std::tuple<ScalarTs...> &scalar) {
  if constexpr (I == sizeof...(ScalarTs)) {
    return;
  } else {
    auto elem = std::get<I>(scalar);
    auto casted_buf = reinterpret_cast<decltype(elem)*>(buf);
    *casted_buf = elem;
    fill_buff<I + 1, ScalarTs...>((char*)(casted_buf + 1), scalar);
  }
}

template<typename... ScalarTs>
void fill_buff(char *buf, const std::tuple<ScalarTs...> &scalar) {
  fill_buff<0, ScalarTs...>(buf, scalar);
}

template<size_t I, typename ...ScalarTs>
void fill_tuple(char *buf, std::tuple<ScalarTs...> &tuple) {
  if constexpr (I == sizeof...(ScalarTs)) {
    return;
  } else {
    typedef std::remove_reference_t<decltype(std::get<I>(tuple))> elemT;
    auto casted_buf = reinterpret_cast<elemT*>(buf);
    std::get<I>(tuple) = *casted_buf;
    fill_tuple<I + 1, ScalarTs...>(reinterpret_cast<char *>(casted_buf + 1), tuple);
  }
}

template<typename ...ScalarTs>
void fill_tuple(char *buf, std::tuple<ScalarTs...> &tuple) {
  fill_tuple<0, ScalarTs...>(buf, tuple);
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
std::vector<PointCoord> CudaPointCloud<ScalarTs...>::GetHostPoints() const {
  std::vector<PointCoord> host_pts(pcl_size_);
  size_t mem_required = pcl_size_ * sizeof(PointCoord);
  cudaThrowIfStatusNotOk(cudaMemcpy(host_pts.data(), xyz_ptr_, mem_required,
                                    cudaMemcpyDeviceToHost));
  return host_pts;
}

template<typename... ScalarTs>
std::vector<std::tuple<ScalarTs...>> CudaPointCloud<ScalarTs...>::GetHostScalars() const
requires HAS_SCALARS_ {
  size_t row_size = std::reduce(scalar_sizes_.begin(), scalar_sizes_.end());
  std::vector<std::tuple<ScalarTs...>> scalars;

  std::vector<char> host_buf(row_size);
  for (size_t i = 0; i < pcl_size_; i++) {
    void *scalar_dev_ptr = reinterpret_cast<char *>(scalar_ptr_) + i * row_size;
    cudaThrowIfStatusNotOk(cudaMemcpy(host_buf.data(), scalar_dev_ptr, row_size,
                                      cudaMemcpyDeviceToHost));
    std::tuple<ScalarTs...> scalar;
    fill_tuple<ScalarTs...>(host_buf.data(), scalar);
    scalars.push_back(scalar);
  }

  return scalars;
}

template<typename... ScalarTs>
void CudaPointCloud<ScalarTs...>::resize(size_t n) {
  PointCoord *xyz_copy;
  size_t new_n_bytes = n * sizeof(PointCoord);
  size_t curr_n_bytes = pcl_size_ * sizeof(PointCoord);
  auto n_bytes_to_copy = std::min(curr_n_bytes, new_n_bytes);

  cudaThrowIfStatusNotOk(cudaMalloc(reinterpret_cast<void **>(&xyz_copy), n_bytes_to_copy));
  cudaThrowIfStatusNotOk(cudaMemcpy(xyz_copy, xyz_ptr_, n_bytes_to_copy, cudaMemcpyDeviceToDevice));

  cudaThrowIfStatusNotOk(cudaFree(xyz_ptr_));
  cudaMalloc(reinterpret_cast<void **>(&xyz_ptr_), new_n_bytes);
  cudaThrowIfStatusNotOk(cudaMemcpy(xyz_ptr_, xyz_copy, n_bytes_to_copy, cudaMemcpyDeviceToDevice));
  cudaThrowIfStatusNotOk(cudaFree(xyz_copy));

  if constexpr (HAS_SCALARS_) {
    void *scalar_copy;
    size_t scalar_row_size = std::reduce(scalar_sizes_.begin(), scalar_sizes_.end());
    new_n_bytes = n * scalar_row_size;
    curr_n_bytes = pcl_size_ * scalar_row_size;
    n_bytes_to_copy = std::min(curr_n_bytes, new_n_bytes);

    cudaThrowIfStatusNotOk(cudaMalloc(&scalar_copy, n_bytes_to_copy));
    cudaThrowIfStatusNotOk(cudaMemcpy(scalar_copy, scalar_ptr_, n_bytes_to_copy,
                                      cudaMemcpyDeviceToDevice));

    cudaThrowIfStatusNotOk(cudaFree(scalar_ptr_));
    cudaThrowIfStatusNotOk(cudaMalloc(&scalar_ptr_, new_n_bytes));
    cudaThrowIfStatusNotOk(cudaMemcpy(scalar_ptr_, scalar_copy, n_bytes_to_copy,
                                      cudaMemcpyDeviceToDevice));
    cudaThrowIfStatusNotOk(cudaFree(scalar_copy));
  }

  pcl_size_ = n;
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
  cudaThrowIfStatusNotOk(cudaMalloc(reinterpret_cast<void **>(&xyz_ptr_), mem_size));

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
    fill_buff<ScalarTs...>(buf_start, scalar);
  }

  cudaThrowIfStatusNotOk(cudaMemcpy(scalar_ptr_, host_buf, mem_size, cudaMemcpyHostToDevice));

  free(host_buf);
}

template<typename... ScalarTs>
void CudaPointCloud<ScalarTs...>::cudaThrowIfStatusNotOk(cudaError_t e) const {
  if (e) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
  }
}

INSTANTIATE_CUDA_POINT_CLOUD()
INSTANTIATE_CUDA_POINT_CLOUD(float)
INSTANTIATE_CUDA_POINT_CLOUD(uint8_t, uint8_t, uint8_t)
INSTANTIATE_CUDA_POINT_CLOUD(float, float, float)

}  // namespace cuda_point_cloud
