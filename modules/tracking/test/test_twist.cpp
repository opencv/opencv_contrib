#include "test_precomp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/tracking/twist.hpp"

namespace opencv_test
{
namespace
{

class TwistTest : public ::testing::Test
{
protected:
    cv::detail::tracking::Twist twist;
    cv::Mat uv, depth, K, J, duv;
    cv::Vec6d result;

    void SetUp() override
    {
        uv = (cv::Mat_<float>(2, 2) << 0.0, 0.0, 0.0, 0.0);
        depth = (cv::Mat_<float>(1, 2) << 1.0, 1.0);
        K = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        duv = (cv::Mat_<float>(2, 2) << 0.0, 0.0, 0.0, 0.0);
    }
};

TEST_F(TwistTest, TestInteractionMatrix)
{
    twist.interactionMatrix(uv, depth, K, J);
    ASSERT_EQ(J.cols, 6);
    ASSERT_EQ(J.rows, 4);
}

TEST_F(TwistTest, TestCompute)
{
    result = twist.compute(uv, duv, depth, K);
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(result[i], 0.0, 1e-6);
}

} // namespace
} // namespace opencv_test
