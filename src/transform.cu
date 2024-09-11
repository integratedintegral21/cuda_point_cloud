#include "transform.hpp"

#include "utils.hpp"

namespace cuda_point_cloud {


__device__ __host__ inline size_t flatten_idx(int stride, size_t i, size_t j) {
  return i * stride + j;
}

__global__ void transformPclKernel(const PointCoord *in_xyz,
                                   PointCoord *out_xyz,
                                   size_t pcl_size,
                                   const float *transform) {
  // One point per thread
  size_t point_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadIdx.x > pcl_size) return;

  const auto &src_pt = in_xyz[point_i];
  auto &dst_pt = out_xyz[point_i];
  dst_pt.x = src_pt.x * transform[flatten_idx(3, 0, 0)] +
             src_pt.y * transform[flatten_idx(3, 1, 0)] +
             src_pt.z * transform[flatten_idx(3, 2, 0)] +
             transform[flatten_idx(3, 3, 0)];
  dst_pt.y = src_pt.x * transform[flatten_idx(3, 0, 1)] +
             src_pt.y * transform[flatten_idx(3, 1, 1)] +
             src_pt.z * transform[flatten_idx(3, 2, 1)] +
             transform[flatten_idx(3, 3, 1)];
  dst_pt.z = src_pt.x * transform[flatten_idx(3, 0, 2)] +
             src_pt.y * transform[flatten_idx(3, 1, 2)] +
             src_pt.z * transform[flatten_idx(3, 2, 2)] +
             transform[flatten_idx(3, 3, 2)];
}

template <typename ...ScalarTs>
void transformPointCloud(const CudaPointCloud<ScalarTs...> &in_pcl,
                         CudaPointCloud<ScalarTs...> &out_pcl,
                         Eigen::Matrix<float, 3, 4, Eigen::ColMajor> &transform) {
  size_t pcl_size = in_pcl.Size();
  if (pcl_size == 0) {
    return;
  }

  if (out_pcl.Size() != pcl_size) {
    out_pcl.Resize(pcl_size);
  }

  float *transform_dev;
  size_t trans_n_bytes = transform.size() * sizeof(float);
  cudaThrowIfStatusNotOk(cudaMalloc(reinterpret_cast<void **>(&transform_dev), trans_n_bytes));
  cudaThrowIfStatusNotOk(cudaMemcpy(transform_dev, transform.data(), trans_n_bytes,
                                    cudaMemcpyHostToDevice));
  const PointCoord *in_xyz = in_pcl.PointCoordDevPtr();
  PointCoord *out_xyz = out_pcl.PointCoordDevPtr();

  size_t block_size = 128;
  size_t max_threads_per_grid = block_size * MAX_N_BLOCKS;

  // In case the point cloud is larger than the maximum number of threads
  for (int i = 0; i * max_threads_per_grid < pcl_size; i++) {
    size_t start_idx = i * max_threads_per_grid;
    size_t chunk_size = std::min(max_threads_per_grid, pcl_size - start_idx);

    size_t n_blocks = std::ceil(static_cast<float>(chunk_size) / block_size);
    transformPclKernel<<<n_blocks, block_size>>>(in_xyz + start_idx,
                                                 out_xyz + start_idx,
                                                 chunk_size,
                                                 transform_dev);
  }
}

INSTANTIATE_TRANSFORM();
INSTANTIATE_TRANSFORM(float);
INSTANTIATE_TRANSFORM(uint8_t, uint8_t, uint8_t);
INSTANTIATE_TRANSFORM(float, float, float);

}  // namespace cuda_point_cloud
