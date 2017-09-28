// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <bitset>

using namespace cv;

class CV_AverageHashTest : public cvtest::BaseTest
{
public:
    CV_AverageHashTest();
    ~CV_AverageHashTest();
protected:
    void run(int /* idx */);
};

CV_AverageHashTest::CV_AverageHashTest(){}
CV_AverageHashTest::~CV_AverageHashTest(){}

void CV_AverageHashTest::run(int )
{
    cv::Mat const input = (cv::Mat_<uchar>(8, 8) <<
                           1, 5, 4, 6, 3, 2, 7, 8,
                           2, 4, 8, 9, 2, 1, 4, 3,
                           3, 4, 5, 7, 9, 8, 7, 6,
                           1, 2, 3, 4, 5, 6, 7, 8,
                           8, 7, 2, 3, 6, 4, 5, 1,
                           3, 4, 1, 2, 9, 8, 4, 2,
                           6, 7, 8, 9, 7, 4, 3, 2,
                           8, 7, 6, 5, 4, 3, 2, 1);
    cv::Mat hash;
    cv::img_hash::averageHash(input, hash);
    bool const expectResult[] =
    {
        0,0,0,1,0,0,1,1,
        0,0,1,1,0,0,0,0,
        0,0,0,1,1,1,1,1,
        0,0,0,0,0,1,1,1,
        1,1,0,0,1,0,0,0,
        0,0,0,0,1,1,0,0,
        1,1,1,1,1,0,0,0,
        1,1,1,0,0,0,0,0
    };
    uchar const *hashPtr = hash.ptr<uchar>(0);
    for(int i = 0; i != hash.cols; ++i)
    {
        std::bitset<8> const bits = hashPtr[i];
        for(int j = 0; j != 8; ++j)
        {
            EXPECT_EQ(bits[j], expectResult[i*8+j]);
        }
    }
}

TEST(average_hash_test, accuracy) { CV_AverageHashTest test; test.safe_run(); }
