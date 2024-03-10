#include "test_precomp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/tracking/twist.hpp"

#define eps 1e-4

namespace opencv_test
{
namespace
{

class TwistTest : public ::testing::Test
{
protected:
    cv::detail::tracking::Twist twist;
    cv::Mat K, J;

    void SetUp() override
    {
        K = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
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

    cv::Mat uv = (cv::Mat_<float>(2, 1) << 1.0, 1.0);
    cv::Mat depth = (cv::Mat_<float>(1, 1) << 2.0);

    twist.interactionMatrix(uv, depth, K, J);
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
    cv::Mat uv = (cv::Mat_<float>(2, 2) << 0.0, 0.0, 0.0, 0.0);
    cv::Mat depth = (cv::Mat_<float>(1, 2) << 1.0, 1.0);
    cv::Mat duv = (cv::Mat_<float>(4, 1) << 0.0, 0.0, 0.0, 0.0);

    cv::Vec6d result = twist.compute(uv, duv, depth, K);
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

    cv::Mat uv = (cv::Mat_<float>(2, 3) << 1.0, 2.0, 3.0, 1.0, 2.0, 3.0);
    cv::Mat depth = (cv::Mat_<float>(1, 3) << 1.0, 2.0, 3.0);
    cv::Mat duv = (cv::Mat_<float>(6, 1) << 1.0, 2.0, 1.0, 3.0, 1.0, 4.0);

    twist.interactionMatrix(uv, depth, K, J);
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

    cv::Vec6d result = twist.compute(uv, duv, depth, K);
    float expected_twist[6] = {0.5f, 0.5f, 1.875f, 0.041667f, -0.041667f, -0.5f};
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(result[i], expected_twist[i], eps);
}

} // namespace
} // namespace opencv_test
