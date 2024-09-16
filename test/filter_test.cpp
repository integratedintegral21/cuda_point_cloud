#include <memory>

#include <gtest/gtest.h>

#include "filter.hpp"

using namespace cuda_point_cloud;

struct FilterFixture: ::testing::Test {
  std::shared_ptr<CudaPointCloudXYZI> small_pcl;
  std::shared_ptr<CudaPointCloudXYZI> medium_pcl;
  std::shared_ptr<CudaPointCloudXYZI> large_pcl;

  static std::tuple<float, float> getUnNormCoeffs(float min, float max, size_t size) {
    return {(max - min) / (size - 1), min};
  }

  void InitPcl(std::shared_ptr<CudaPointCloudXYZI> *pcl,
               size_t pcl_size,
               float min_x, float max_x,
               float min_y, float max_y,
               float min_z, float max_z) {
    std::vector<PointCoord> xyz(pcl_size);
    std::vector<std::tuple<float>> intensity(pcl_size);

    auto [scale_x, offset_x] = getUnNormCoeffs(min_x, max_x, pcl_size);
    auto [scale_y, offset_y] = getUnNormCoeffs(min_y, max_y, pcl_size);
    auto [scale_z, offset_z] = getUnNormCoeffs(min_z, max_z, pcl_size);
    float scale_i = 255. / pcl_size;
    for (size_t i = 0; i < pcl_size; i++) {
      float x = i * scale_x + offset_x;
      float y = i * scale_y + offset_y;
      float z = i * scale_z + offset_z;
      float scalar = i * scale_i;
      xyz[i].x = x;
      xyz[i].y = y;
      xyz[i].z = z;
      intensity[i] = scalar;
    }
    *pcl = std::make_shared<CudaPointCloudXYZI>(xyz, intensity);
  }

  FilterFixture() {
    constexpr size_t small_size = 1024;
    constexpr size_t medium_size = 96000;
    constexpr size_t large_size = 33554432;  // 2^25

    constexpr float min_x = -1000;
    constexpr float max_x = 1000;

    constexpr float min_y = -2000;
    constexpr float max_y = 2000;

    constexpr float min_z = -3000;
    constexpr float max_z = 3000;
    InitPcl(&small_pcl, small_size, min_x, max_x, min_y, max_y, min_z, max_z);
    InitPcl(&medium_pcl, medium_size, min_x, max_x, min_y, max_y, min_z, max_z);
//    InitPcl(&large_pcl, large_size, min_x, max_x, min_y, max_y, min_z, max_z);
  }
};

TEST_F(FilterFixture, GenericFilter) {
  CudaPointCloudXYZI out_pcl;
  auto filter_fun = [] __host__ __device__ (const PointCoord &pt) {
    return pt.x >= 0;
  };
  filter_by_coordinates(*small_pcl, out_pcl, filter_fun);

}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
