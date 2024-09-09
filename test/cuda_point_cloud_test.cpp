#include <gtest/gtest.h>

#include "PointCloud.hpp"

using namespace cuda_point_cloud;

struct PointCloudFixture: ::testing::Test {
  std::vector<cuda_point_cloud::PointCoord> xyz;

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
  }

  void TearDown() override {
    Test::TearDown();
  }
};

TEST_F(PointCloudFixture, ConstructorTest) {
  ASSERT_NO_THROW(CudaPointCloudXYZ p(xyz));
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}