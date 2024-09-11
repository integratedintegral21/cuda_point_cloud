#include <numeric>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "point_cloud.hpp"
#include "transform.hpp"

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
  ASSERT_NO_THROW(CudaPointCloudXYZ p);
  ASSERT_NO_THROW(CudaPointCloudXYZRGB p({}, {}));
  ASSERT_THROW(CudaPointCloudXYZRGB p(xyz, {{0, 0, 0}, {1, 1, 1}}), std::invalid_argument);

  CudaPointCloudXYZ pcl(xyz);
  ASSERT_EQ(pcl.Size(), xyz.size());

  CudaPointCloudXYZ empty_pcl;
  ASSERT_EQ(empty_pcl.Size(), 0);

  CudaPointCloudXYZRGB pcl_rgb(xyz, rgb);
  auto scalar_sizes = pcl_rgb.ScalarSizes();
  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(scalar_sizes[i], 1);
  }

  // Make sure points are set correctly
  PointCoord *test_xyz_dev = pcl_rgb.PointCoordDevPtr();
  size_t pcl_size = pcl_rgb.Size();
  size_t mem_required = pcl_size * sizeof(PointCoord);
  std::vector<PointCoord> test_xyz(pcl_size);

  if (auto status = cudaMemcpy(test_xyz.data(), test_xyz_dev, mem_required, cudaMemcpyDeviceToHost)) {
    FAIL();
  }

  for (size_t i = 0; i < pcl_size; i++) {
    const auto pt = test_xyz[i];
    const auto ref_pt = xyz[i];
    ASSERT_EQ(pt.x, ref_pt.x);
    ASSERT_EQ(pt.y, ref_pt.y);
    ASSERT_EQ(pt.z, ref_pt.z);
  }

  // Check scalars as well
  char *test_rgb_dev = (char *)pcl_rgb.ScalarDevPtr();
  mem_required = pcl_size * std::reduce(scalar_sizes.begin(), scalar_sizes.end());
  std::vector<char> test_rgb(mem_required);  // A byte-array
  if (auto status = cudaMemcpy(test_rgb.data(), test_rgb_dev, mem_required, cudaMemcpyDeviceToHost)) {
    FAIL();
  }

  for (size_t i = 0; i < pcl_size; i++) {
    auto r = static_cast<uint8_t>(test_rgb[3 * i]);
    auto g = static_cast<uint8_t>(test_rgb[3 * i + 1]);
    auto b = static_cast<uint8_t>(test_rgb[3 * i + 2]);
    auto ref_r = std::get<0>(rgb[i]);
    auto ref_g = std::get<1>(rgb[i]);
    auto ref_b = std::get<2>(rgb[i]);
    ASSERT_EQ(r, ref_r);
    ASSERT_EQ(g, ref_g);
    ASSERT_EQ(b, ref_b);
  }
}

TEST_F(PointCloudFixture, HostGettersTest) {
  CudaPointCloudXYZRGB pcl_rgb(xyz, rgb);

  auto host_xyz = pcl_rgb.GetHostPoints();
  size_t pcl_size = pcl_rgb.Size();
  for (size_t i = 0; i < pcl_size; i++) {
    const auto &pt = host_xyz[i];
    ASSERT_EQ(pt.x, xyz[i].x);
    ASSERT_EQ(pt.y, xyz[i].y);
    ASSERT_EQ(pt.z, xyz[i].z);
  }

  auto host_rgb = pcl_rgb.GetHostScalars();
  for (size_t i = 0; i < pcl_size; i++) {
    auto r = std::get<0>(host_rgb[i]);
    auto g = std::get<1>(host_rgb[i]);
    auto b = std::get<2>(host_rgb[i]);
    ASSERT_EQ(r, std::get<0>(rgb[i]));
    ASSERT_EQ(g, std::get<1>(rgb[i]));
    ASSERT_EQ(b, std::get<2>(rgb[i]));
  }
}

TEST_F(PointCloudFixture, ResizeTest) {
  CudaPointCloudXYZRGB pcl_rgb(xyz, rgb);

  pcl_rgb.Resize(3);
  ASSERT_EQ(pcl_rgb.Size(), 3);

  size_t pcl_size = pcl_rgb.Size();
  auto host_xyz = pcl_rgb.GetHostPoints();
  auto host_rgb = pcl_rgb.GetHostScalars();
  for (size_t i = 0; i < pcl_size; i++) {
    const auto &pt = host_xyz[i];
    ASSERT_EQ(pt.x, xyz[i].x);
    ASSERT_EQ(pt.y, xyz[i].y);
    ASSERT_EQ(pt.z, xyz[i].z);
  }
  for (size_t i = 0; i < pcl_size; i++) {
    auto r = std::get<0>(host_rgb[i]);
    auto g = std::get<1>(host_rgb[i]);
    auto b = std::get<2>(host_rgb[i]);
    ASSERT_EQ(r, std::get<0>(rgb[i]));
    ASSERT_EQ(g, std::get<1>(rgb[i]));
    ASSERT_EQ(b, std::get<2>(rgb[i]));
  }

  CudaPointCloudXYZRGB empty_pcl;
  empty_pcl.Resize(5);

  ASSERT_EQ(empty_pcl.Size(), 5);
  ASSERT_EQ(empty_pcl.GetHostPoints().size(), 5);
  ASSERT_EQ(empty_pcl.GetHostScalars().size(), 5);
  ASSERT_NE(empty_pcl.PointCoordDevPtr(), nullptr);
  ASSERT_NE(empty_pcl.ScalarDevPtr(), nullptr);

  // Set data
  pcl_size = empty_pcl.Size();
  size_t n_bytes_to_copy = pcl_size * sizeof(PointCoord);
  if (cudaMemcpy(empty_pcl.PointCoordDevPtr(), xyz.data(), n_bytes_to_copy, cudaMemcpyHostToDevice)) {
    FAIL();
  }
  host_xyz = empty_pcl.GetHostPoints();
  for (size_t i = 0; i < pcl_size; i++) {
    ASSERT_EQ(host_xyz[i].x, xyz[i].x);
  }

  // now check scalars
  n_bytes_to_copy = pcl_size * 3;
  std::vector<unsigned char> rgb_buf(n_bytes_to_copy);
  for (size_t i = 0; i < pcl_size; i++) {
    rgb_buf[3 * i] = std::get<0>(rgb[i]);
    rgb_buf[3 * i + 1] = std::get<1>(rgb[i]);
    rgb_buf[3 * i + 2] = std::get<2>(rgb[i]);
  }
  if (cudaMemcpy(empty_pcl.ScalarDevPtr(), rgb_buf.data(), n_bytes_to_copy, cudaMemcpyHostToDevice)) {
    FAIL();
  }
  host_rgb = empty_pcl.GetHostScalars();
  ASSERT_EQ(host_rgb.size(), pcl_size);
  for (size_t i = 0; i < pcl_size; i++) {
    ASSERT_EQ(host_rgb[i], rgb[i]);
  }
}

TEST_F(PointCloudFixture, TransformTest) {
  CudaPointCloudXYZRGB pcl_rgb(xyz, rgb);
  CudaPointCloudXYZRGB pcl_rgb_out;
  Eigen::Matrix<float, 3, 4> identity;
  identity << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0;

  transformPointCloud(pcl_rgb, pcl_rgb_out, identity);

  auto transformed_xyz = pcl_rgb_out.GetHostPoints();
  size_t pcl_size = pcl_rgb_out.Size();
  ASSERT_EQ(pcl_size, pcl_rgb.Size());
  for (size_t i = 0; i < pcl_size; i++) {
    ASSERT_EQ(transformed_xyz[i].x, xyz[i].x);
    ASSERT_EQ(transformed_xyz[i].y, xyz[i].y);
    ASSERT_EQ(transformed_xyz[i].z, xyz[i].z);
  }

  std::vector<PointCoord> xyz_big;
  pcl_size = 1 << 21;
  for (size_t i = 0; i < pcl_size; i++) {
    float x = 2048. / pcl_size * i;
    float y = 1024. / pcl_size * i + 200;
    float z = 2048. / pcl_size * i - 5000;
    xyz_big.push_back({x, y, z});
  }

  CudaPointCloudXYZ pcl_big(xyz_big);
  CudaPointCloudXYZ pcl_big_out;

  transformPointCloud(pcl_big, pcl_big_out, identity);
  transformed_xyz = pcl_big_out.GetHostPoints();
  for (size_t i = 0; i < pcl_size; i++) {
    ASSERT_EQ(transformed_xyz[i].x, xyz_big[i].x);
    ASSERT_EQ(transformed_xyz[i].y, xyz_big[i].y);
    ASSERT_EQ(transformed_xyz[i].z, xyz_big[i].z);
  }

  Eigen::Matrix<float, 3, 4> translation_only;
  translation_only << 1, 0, 0, 21.5,
                      0, 1, 0, -24.1,
                      0, 0, 1, 33.3;

  transformPointCloud(pcl_big, pcl_big_out, translation_only);
  transformed_xyz = pcl_big_out.GetHostPoints();
  for (size_t i = 0; i < pcl_size; i++) {
    ASSERT_NEAR(transformed_xyz[i].x, xyz_big[i].x + 21.5, 0.01);
    ASSERT_NEAR(transformed_xyz[i].y, xyz_big[i].y - 24.1, 0.01);
    ASSERT_NEAR(transformed_xyz[i].z, xyz_big[i].z + 33.3, 0.01);
  }
}

TEST_F(PointCloudFixture, LargeTransform) {
  size_t pcl_size = 1 << 25;
  std::vector<PointCoord> xyz;
  for (size_t i = 0; i < pcl_size; i++) {
    float x = 2048. / pcl_size * i;
    float y = 1024. / pcl_size * i + 200;
    float z = 2048. / pcl_size * i - 5000;
    xyz.push_back({x, y, z});
  }

  CudaPointCloudXYZ pcl(xyz);
  CudaPointCloudXYZ pcl_out;
  Eigen::Matrix<float, 3, 4> transform;
  transform << 0, 1, 0, 21.5,
               0, 0, 1, -24.1,
               1, 0, 0, 33.3;
  transformPointCloud(pcl, pcl_out, transform);
  auto transformed_xyz = pcl_out.GetHostPoints();
  for (size_t i = 0; i < pcl_size; i++) {
    ASSERT_NEAR(transformed_xyz[i].x, xyz[i].y + 21.5, 0.01);
    ASSERT_NEAR(transformed_xyz[i].y, xyz[i].z - 24.1, 0.01);
    ASSERT_NEAR(transformed_xyz[i].z, xyz[i].x + 33.3, 0.01);
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}