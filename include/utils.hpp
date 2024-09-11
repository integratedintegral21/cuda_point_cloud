#ifndef CUDA_POINT_CLOUD_UTILS_HPP
#define CUDA_POINT_CLOUD_UTILS_HPP

#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#define MAX_N_BLOCKS (1 << 16)  // 65536

inline void cudaThrowIfStatusNotOk(cudaError_t e) {
  if (e) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
  }
}

#endif  // CUDA_POINT_CLOUD_UTILS_HPP
