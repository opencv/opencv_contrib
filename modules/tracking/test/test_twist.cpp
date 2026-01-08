#include "test_precomp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/tracking/twist.hpp"

namespace opencv_test
{
namespace
{

using namespace cv::detail::tracking;

float const eps = 1e-4f;

class TwistTest : public ::testing::Test
{
protected:
    cv::Mat J, K;

    void SetUp() override
    {
        cv::Matx33f K = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        this->K = cv::Mat(K);
    }
};

TEST_F(TwistTest, TestInteractionMatrix)
{
    // import machinevisiontoolbox as mv
    // cam = mv.CentralCamera()
    // print(cam.K)
    // print(cam.visjac_p([1, 1], 2.0))
    // [[1. 0. 0.]
    //  [0. 1. 0.]
    //  [0. 0. 1.]]
    // [[-0.5  0.   0.5  1.  -2.   1. ]
    //  [ 0.  -0.5  0.5  2.  -1.  -1. ]]

    cv::Mat uv = cv::Mat(2, 1, CV_32F, {1.0f, 1.0f});
    cv::Mat depth = cv::Mat(1, 1, CV_32F, {2.0f});

    computeInteractionMatrix(uv, depth, K, J);
    ASSERT_EQ(J.cols, 6);
    ASSERT_EQ(J.rows, 2);
    float expected[2][6] = {{-0.5f, 0.0f, 0.5f, 1.0f, -2.0f, 1.0f},
                            {0.0f, -0.5f, 0.5f, 2.0f, -1.0f, -1.0f}};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 6; j++)
            ASSERT_NEAR(J.at<float>(i, j), expected[i][j], eps);
}

TEST_F(TwistTest, TestComputeWithZeroPixelVelocities)
{
    cv::Mat uv = cv::Mat(2, 2, CV_32F, {1.0f, 0.0f, 3.0f, 0.0f});
    cv::Mat depths = cv::Mat(1, 2, CV_32F, {1.1f, 1.0f});
    cv::Mat duv = cv::Mat(4, 1, CV_32F, {0.0f, 0.0f, 0.0f, 0.0f});

    cv::Vec6d result = computeTwist(uv, duv, depths, K);
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(result[i], 0.0, eps);
}

TEST_F(TwistTest, TestComputeWithNonZeroPixelVelocities)
{
    // import machinevisiontoolbox as mv
    // cam = mv.CentralCamera()
    // pixels = np.array([[1, 2, 3],
    //                    [1, 2, 3]], dtype=float)
    // depths = np.array([1.0, 2.0, 3.0])
    // Jac = cam.visjac_p(pixels, depths)
    // duv = np.array([1, 2, 1, 3, 1, 4])
    // twist = np.linalg.lstsq(Jac, duv, rcond=None)[0]
    // print(twist)
    // print(Jac)
    // [ 0.5       0.5       1.875     0.041667 -0.041667 -0.5     ]
    // [[ -1.         0.         1.         1.        -2.         1.      ]
    //  [  0.        -1.         1.         2.        -1.        -1.      ]
    //  [ -0.5        0.         1.         4.        -5.         2.      ]
    //  [  0.        -0.5        1.         5.        -4.        -2.      ]
    //  [ -0.333333   0.         1.         9.       -10.         3.      ]
    //  [  0.        -0.333333   1.        10.        -9.        -3.      ]]

    float uv_data[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    cv::Mat uv = cv::Mat(2, 3, CV_32F, uv_data);
    float depth_data[] = {1.0f, 2.0f, 3.0f};
    cv::Mat depth = cv::Mat(1, 3, CV_32F, depth_data);
    float duv_data[] = {1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f};
    cv::Mat duv = cv::Mat(6, 1, CV_32F, duv_data);

    computeInteractionMatrix(uv, depth, K, J);
    ASSERT_EQ(J.cols, 6);
    ASSERT_EQ(J.rows, 6);
    float expected_jac[6][6] = {{-1.0f, 0.0f, 1.0f, 1.0f, -2.0f, 1.0f},
                                {0.0f, -1.0f, 1.0f, 2.0f, -1.0f, -1.0f},
                                {-0.5f, 0.0f, 1.0f, 4.0f, -5.0f, 2.0f},
                                {0.0f, -0.5f, 1.0f, 5.0f, -4.0f, -2.0f},
                                {-0.333333f, 0.0f, 1.0f, 9.0f, -10.0f, 3.0f},
                                {0.0f, -0.333333f, 1.0f, 10.0f, -9.0f, -3.0f}};

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            ASSERT_NEAR(J.at<float>(i, j), expected_jac[i][j], eps);

    cv::Vec6d result = computeTwist(uv, duv, depth, K);
    float expected_twist[6] = {0.5f, 0.5f, 1.875f, 0.041667f, -0.041667f, -0.5f};
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(result[i], expected_twist[i], eps);
}

} // namespace
} // namespace opencv_test
