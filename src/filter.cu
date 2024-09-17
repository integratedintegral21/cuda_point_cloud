#include "filter.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "utils.hpp"
#include "iostream"

namespace cg = cooperative_groups;

namespace cuda_point_cloud {

template <typename Group, typename F>
__device__ size_t do_filter(Group &g,
                            const PointCoord *in_chunk,
                            PointCoord *out_chunk,
                            F &&f) {
  size_t point_idx = g.thread_rank();
  size_t is_filtered = f(in_chunk[point_idx]);
  size_t my_start_idx = cg::exclusive_scan(g, is_filtered);
  if (is_filtered) {
    out_chunk[my_start_idx] = in_chunk[point_idx];
  }

  // The last thread in the group holds the number of points after filtering,
  // return it in every thread
  return g.shfl(my_start_idx + is_filtered, g.num_threads() - 1);
}

template <typename F>
__global__ void coord_filter_kernel(const PointCoord *in_pcl,
                                    PointCoord *out_pcl,
                                    size_t pcl_size,
                                    F &&f,
                                    size_t *partial_counts) {
  assert(pcl_size % 128 == 0);
  auto tb = cg::this_thread_block();
  auto filter_tile = cg::tiled_partition<128>(tb);  // One tile per block
  size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t n_threads = gridDim.x * blockDim.x;

  __shared__ PointCoord local_chunk[128];
  __shared__ PointCoord local_result[128];

  size_t n_batches = std::ceil((pcl_size - thread_id) / static_cast<float>(n_threads));

  for (size_t i = 0; i < n_batches; i++) {
    size_t global_idx = i * pcl_size + thread_id;
    local_chunk[threadIdx.x] = in_pcl[global_idx];
    filter_tile.sync();

    size_t n_filtered = do_filter(filter_tile, local_chunk, local_result, f);
    filter_tile.sync();

    partial_counts[blockIdx.x + i * blockDim.x] = n_filtered;
    out_pcl[global_idx] = local_result[threadIdx.x];

    tb.sync();
  };
}

template<typename ...ScalarTs>
void filter_by_coordinates(const CudaPointCloud<ScalarTs...> &in_pcl,
                           CudaPointCloud<ScalarTs...> &out_pcl,
                           std::function<bool(const PointCoord &)> &&filter_fun) {
  size_t pcl_size = in_pcl.Size();
  if (pcl_size == 0) {
    return;
  }

  if (out_pcl.Size() != pcl_size) {
    out_pcl.Resize(pcl_size);
  }

  const PointCoord *in_xyz = in_pcl.PointCoordDevPtr();
  PointCoord *out_xyz = out_pcl.PointCoordDevPtr();

  size_t block_size = 128;
  size_t desired_n_blocks = std::ceil(pcl_size / static_cast<float>(block_size));
  size_t n_blocks = std::min(desired_n_blocks, static_cast<size_t>(MAX_N_BLOCKS));

  size_t *partial_sums;
  cudaThrowIfStatusNotOk(cudaMalloc(&partial_sums, desired_n_blocks * sizeof(size_t)));

  auto gpu_fun = [=] __device__ (const PointCoord &pt) {
    return pt.x > 0 && pt.y > 0 && pt.z > 0;
  };

  coord_filter_kernel<<<n_blocks, block_size>>>(in_xyz, out_xyz, pcl_size, gpu_fun, partial_sums);

  size_t *host_partial_sums = new size_t[desired_n_blocks];
  cudaThrowIfStatusNotOk(cudaMemcpy(host_partial_sums, partial_sums,
                                    desired_n_blocks * sizeof(size_t),
                                    cudaMemcpyDeviceToHost));
  std::cout << "Partial sums: " << std::endl;
  for (size_t i = 0; i < desired_n_blocks; i++) {
    std::cout << host_partial_sums[i] << std::endl;
  }
  delete[] host_partial_sums;
}

INSTANTIATE_FILTER()
INSTANTIATE_FILTER(float)
INSTANTIATE_FILTER(uint8_t, uint8_t, uint8_t)
INSTANTIATE_FILTER(float, float, float)

}  // namespace cuda_point_cloud
