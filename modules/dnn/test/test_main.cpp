#include "test_precomp.hpp"

CV_TEST_MAIN("")

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

TEST(BlobShape_SimpleConstr, Regression)
{
    BlobShape sd;

    BlobShape s1(0);
    EXPECT_EQ(s1.dims(), 1);
    EXPECT_EQ(s1[0], 0);

    BlobShape s2(0, 0);
    EXPECT_EQ(s2.dims(), 2);
    EXPECT_EQ(s2[0], 0);
    EXPECT_EQ(s2[1], 0);
}

TEST(BlobShape_EmptyFill, Regression)
{
    BlobShape s(10, (int*)NULL);
    EXPECT_EQ(s.dims(), 10);
}

}
