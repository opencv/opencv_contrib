// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <bitset>

using namespace cv;

class CV_PHashTest : public cvtest::BaseTest
{
public:
    CV_PHashTest();
    ~CV_PHashTest();
protected:
    void run(int /* idx */);
};

CV_PHashTest::CV_PHashTest(){}
CV_PHashTest::~CV_PHashTest(){}

void CV_PHashTest::run(int )
{
    cv::Mat input(32, 32, CV_8U);
    cv::Mat hash;

    uchar value = 0;
    uchar *inPtr = input.ptr<uchar>(0);
    for(size_t i = 0; i != 32*32; ++i)
    {
        inPtr[i] = value++;
    }

    cv::img_hash::pHash(input, hash);
    bool const expectResult[] =
    {
        1,0,1,1,1,1,1,1,
        0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        0,1,1,1,1,1,1,1,
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

TEST(average_phash_test, accuracy) { CV_PHashTest test; test.safe_run(); }
