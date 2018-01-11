// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace cv;

class CV_MarrHildrethTest : public cvtest::BaseTest
{
public:
    CV_MarrHildrethTest();
    ~CV_MarrHildrethTest();
protected:
    void run(int /* idx */);
};

CV_MarrHildrethTest::CV_MarrHildrethTest(){}
CV_MarrHildrethTest::~CV_MarrHildrethTest(){}

void CV_MarrHildrethTest::run(int )
{
    cv::Mat_<uchar> input(512,512);
    int val = 0;
    for(int row = 0; row != input.rows; ++row)
    {
        for(int col = 0; col != input.cols; ++col)
        {
            input.at<uchar>(row, col) = val % 256;
            ++val;
        }
    }

    cv::Mat hash;
    cv::img_hash::marrHildrethHash(input, hash);
    uchar const expectResult[] =
    {
        252, 126,  63,  31, 143, 199, 227, 241,
        248, 252, 126,  63,  31, 143, 199, 227,
        241, 248, 252, 126,  63,  31, 143, 199,
        227, 241, 248, 252, 126,  63,  31, 143,
        199, 227, 241, 248,  31, 143, 199, 227,
        241, 248, 252, 126,  63, 252, 126,  63,
         31, 143, 199, 227, 241, 248, 252, 126,
         63,  31, 143, 199, 227, 241, 248, 252,
        126,  63,  31, 143, 199, 227, 241, 248
    };
    uchar const *hashPtr = hash.ptr<uchar>(0);
    for(int i = 0; i != 72; ++i)
    {
        if(hashPtr[i] != expectResult[i])
        {
            ts->printf(cvtest::TS::LOG, "Wrong hash value \n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
    }
}

TEST(marr_hildreth_test, accuracy) { CV_MarrHildrethTest test; test.safe_run(); }
