#include <gtest/gtest.h>

#include "PointCloud.hpp"

using namespace cuda_point_cloud;

struct PointCloudFixture: ::testing::Test {
  std::vector<cuda_point_cloud::PointCoord> xyz;
  std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> rgb;

 protected:
  void SetUp() override {
    Test::SetUp();
    xyz = {
        {0, 0, 0},
        {1, 2, 3},
        {4, 5, 6},
        {12, 23, 34},
        {12, 5, 23}
    };

    rgb = {
        {0, 0, 0},
        {128, 128, 64},
        {255, 255, 255},
        {96, 64, 12},
        {1, 1, 145}
    };
  }

  void TearDown() override {
    Test::TearDown();
  }
};

TEST_F(PointCloudFixture, ConstructorTest) {
  ASSERT_NO_THROW(CudaPointCloudXYZ p(xyz));
  ASSERT_NO_THROW(CudaPointCloudXYZRGB p(xyz, rgb));
  ASSERT_NO_THROW(CudaPointCloudXYZ p({}));
  ASSERT_NO_THROW(CudaPointCloudXYZRGB p({}, {}));
  ASSERT_THROW(CudaPointCloudXYZRGB p(xyz, {{0, 0, 0}, {1, 1, 1}}), std::invalid_argument);

  CudaPointCloudXYZ pcl(xyz);
  ASSERT_EQ(pcl.Size(), xyz.size());

  CudaPointCloudXYZ empty_pcl({});
  ASSERT_EQ(empty_pcl.Size(), 0);

  CudaPointCloudXYZRGB pcl_rgb(xyz, rgb);
  auto scalar_sizes = pcl_rgb.ScalarSizes();
  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(scalar_sizes[i], 1);
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}