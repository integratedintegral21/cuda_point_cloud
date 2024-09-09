#include <gtest/gtest.h>

TEST(MyTestSuite, MyTestCase)
{
  int x = 42;
  EXPECT_EQ(x, 42);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}