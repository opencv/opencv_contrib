// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace cv;
using namespace cv::img_hash;


/**
 *The expected results of this test case are come from the phash library,
 *I use it as golden model
 */
class CV_RadialVarianceHashTest : public cvtest::BaseTest
{
public:
    CV_RadialVarianceHashTest();
protected:
    void run(int /* idx */);

    //this test case do not use the original "golden data"
    //of pHash library, I add a small value to nb_pixels in
    //the function "ph_feature_vector" to avoid NaN value
    void testComputeHash();
    void testFeatures();
    void testHash();
    void testPixPerLine();
    void testProjection();

    cv::Mat input;
    Ptr<cv::img_hash::RadialVarianceHash> rvh;
};

CV_RadialVarianceHashTest::CV_RadialVarianceHashTest()
{
    input.create(8, 8, CV_8U);
    uchar *inPtr = input.ptr<uchar>(0);
    for(size_t i = 0; i != input.total(); ++i)
    {
        inPtr[i] = static_cast<uchar>(i);
    }
    rvh = RadialVarianceHash::create(1, 10);
}

void CV_RadialVarianceHashTest::testComputeHash()
{
    cv::Mat hashOne(1, 40, CV_8U);
    uchar buffer[] =
    {
      52,  41,  49,  64,  40,  67,  76,  71,  69,
      55,  58,  68,  72,  78,  63,  73,  66,  77,
      60,  57,  48,  59,  62,  74,  70,  47,  46,
      51,  45,  44,  42,  61,  54,  75,  50,  79,
      65,  43,  53,  56
    };
    cv::Mat hashTwo(1, 40, CV_8U, buffer);
    for(uchar i = 0; i != 40; ++i)
    {
      hashOne.at<uchar>(0, i) = i;
    }

    double const actual = rvh->compare(hashOne, hashTwo);
    ASSERT_NEAR(0.481051, actual, 0.0001);
}

void CV_RadialVarianceHashTest::testFeatures()
{
    std::vector<double> const &features = rvh->getFeatures();
    double const expectResult[] =
    {-1.35784,-0.42703,0.908487,-1.39327,1.17313,
     1.47515,-0.0156121,0.774335,-0.116755,-1.02059};
    for(size_t i = 0; i != features.size(); ++i)
    {
        ASSERT_NEAR(features[i], expectResult[i], 0.0001);
    }
}

void CV_RadialVarianceHashTest::testHash()
{
    cv::Mat const hash = rvh->getHash();
    uchar const expectResult[] =
    {
        127,  92,   0, 158, 101,
         88,  14, 136, 227, 160,
        127,  94,  27, 118, 240,
        166, 153,  96, 254, 162,
        127, 162, 255,  96, 153,
        166, 240, 118,  27,  94,
        127, 160, 227, 136,  14,
         88, 101, 158,   0,  92
    };
    for(int i = 0; i != hash.cols; ++i)
    {
        EXPECT_EQ(hash.at<uchar>(0, i), expectResult[i]);
    }
}

void CV_RadialVarianceHashTest::testPixPerLine()
{
  cv::Mat const pixPerLine = rvh->getPixPerLine(input);
  uchar const expectResult[] =
  {
    8,8,8,0,8,15,7,5,8,8,
  };
  bool const equal =
          std::equal(expectResult, expectResult + pixPerLine.total(),
          pixPerLine.ptr<int>(0));
  if(equal == false)
  {
    ts->printf(cvtest::TS::LOG, "Wrong pixel per line value \n");
    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
  }
}

void CV_RadialVarianceHashTest::testProjection()
{
    cv::Mat const proj = rvh->getProjection();
    uchar const expectResult[] =
    {
      32,  33,  34,  35,  36,  37,  38,  39,
      16,  17,  18,  27,  36,  37,  46,  47,
       0,   9,  18,  19,  36,  45,  46,  55,
       0,   0,   0,   0,   0,   0,   0,   0,
       2,  10,  18,  27,  36,  44,  53,  61,
       4,  59,  51,  44,  36,  29,  22,  14,
       0,  58,  51,  43,  36,  30,  22,  15,
       0,   0,  58,  43,  36,  21,   6,   0,
      56,  49,  42,  43,  36,  21,  22,  15,
      40,  41,  42,  35,  36,  29,  22,  23
    };
    bool const equal =
            std::equal(expectResult, expectResult + proj.total(),
                       proj.ptr<uchar>(0));
    if(equal == false)
    {
      ts->printf(cvtest::TS::LOG, "Wrong projection value \n");
      ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
    }
}

void CV_RadialVarianceHashTest::run(int)
{
    testPixPerLine();
    testProjection();
    testFeatures();
    testComputeHash();
}

TEST(radial_variance_hash_test, accuracy) { CV_RadialVarianceHashTest test; test.safe_run(); }
