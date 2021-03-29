/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {
const string FEATURES2D_DIR = "features2d";
const string DETECTOR_DIR = FEATURES2D_DIR + "/feature_detectors";
const string DESCRIPTOR_DIR = FEATURES2D_DIR + "/descriptor_extractors";
const string IMAGE_FILENAME = "tsukuba.png";
}} // namespace

#include "features2d/test/test_detectors_regression.impl.hpp"
#include "features2d/test/test_descriptors_regression.impl.hpp"

namespace opencv_test { namespace {

#ifdef OPENCV_ENABLE_NONFREE
TEST( Features2d_Detector_SURF, regression )
{
    CV_FeatureDetectorTest test( "detector-surf", SURF::create() );
    test.safe_run();
}
#endif

TEST( Features2d_Detector_STAR, regression )
{
    CV_FeatureDetectorTest test( "detector-star", StarDetector::create() );
    test.safe_run();
}

TEST( Features2d_Detector_Harris_Laplace, regression )
{
    CV_FeatureDetectorTest test( "detector-harris-laplace", HarrisLaplaceFeatureDetector::create() );
    test.safe_run();
}

TEST( Features2d_Detector_Harris_Laplace_Affine_Keypoint_Invariance, regression )
{
    CV_FeatureDetectorTest test( "detector-harris-laplace", AffineFeature2D::create(HarrisLaplaceFeatureDetector::create()));
    test.safe_run();
}

TEST( Features2d_Detector_Harris_Laplace_Affine, regression )
{
    CV_FeatureDetectorTest test( "detector-harris-laplace-affine", AffineFeature2D::create(HarrisLaplaceFeatureDetector::create()));
    test.safe_run();
}

TEST(Features2d_Detector_TBMR_Affine, regression)
{
    CV_FeatureDetectorTest test("detector-tbmr-affine", TBMR::create());
    test.safe_run();
}

/*
 * Descriptors
 */

#ifdef OPENCV_ENABLE_NONFREE
TEST( Features2d_DescriptorExtractor_SURF, regression )
{
#ifdef HAVE_OPENCL
    bool useOCL = cv::ocl::useOpenCL();
    cv::ocl::setUseOpenCL(false);
#endif

    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-surf",  0.05f,
                                                SURF::create() );
    test.safe_run();

#ifdef HAVE_OPENCL
    cv::ocl::setUseOpenCL(useOCL);
#endif
}

#ifdef HAVE_OPENCL
TEST( Features2d_DescriptorExtractor_SURF_OCL, regression )
{
    bool useOCL = cv::ocl::useOpenCL();
    cv::ocl::setUseOpenCL(true);
    if(cv::ocl::useOpenCL())
    {
        CV_DescriptorExtractorTest<L2<float> > test( "descriptor-surf_ocl",  0.05f,
                                                    SURF::create() );
        test.safe_run();
    }
    cv::ocl::setUseOpenCL(useOCL);
}
#endif
#endif // NONFREE

TEST( Features2d_DescriptorExtractor_DAISY, regression )
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-daisy",  0.05f,
                                                DAISY::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_FREAK, regression )
{
    CV_DescriptorExtractorTest<Hamming> test("descriptor-freak", (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                             FREAK::create());
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BRIEF, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-brief",  1,
                                             BriefDescriptorExtractor::create() );
    test.safe_run();
}

template <int threshold = 0>
struct LUCIDEqualityDistance
{
    typedef unsigned char ValueType;
    typedef int ResultType;

    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const
    {
        int res = 0;
        for (int i = 0; i < size; i++)
        {
            if (threshold == 0)
                res += (a[i] != b[i]) ? 1 : 0;
            else
                res += abs(a[i] - b[i]) > threshold ? 1 : 0;
        }
        return res;
    }
};

TEST( Features2d_DescriptorExtractor_LUCID, regression )
{
    CV_DescriptorExtractorTest< LUCIDEqualityDistance<1/*used blur is not bit-exact*/> > test(
            "descriptor-lucid", 2,
            LUCID::create(1, 2)
    );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_LATCH, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-latch",  1,
                                             LATCH::create(32, true, 3, 0) );
    test.safe_run();
}

TEST(Features2d_DescriptorExtractor_BEBLID, regression )
{
    CV_DescriptorExtractorTest<Hamming> test("descriptor-beblid", 1,
                                             BEBLID::create(6.75));
    test.safe_run();
}

#ifdef OPENCV_XFEATURES2D_HAS_VGG_DATA
TEST( Features2d_DescriptorExtractor_VGG, regression )
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-vgg",  0.03f,
                                             VGG::create() );
    test.safe_run();
}
#endif // OPENCV_XFEATURES2D_HAS_VGG_DATA

#ifdef OPENCV_XFEATURES2D_HAS_BOOST_DATA
TEST( Features2d_DescriptorExtractor_BGM, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-boostdesc-bgm",
                                            (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                            BoostDesc::create(BoostDesc::BGM) );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BGM_HARD, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-boostdesc-bgm_hard",
                                            (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                            BoostDesc::create(BoostDesc::BGM_HARD) );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BGM_BILINEAR, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-boostdesc-bgm_bilinear",
                                            (CV_DescriptorExtractorTest<Hamming>::DistanceType)15.f,
                                            BoostDesc::create(BoostDesc::BGM_BILINEAR) );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_LBGM, regression )
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-boostdesc-lbgm",
                                           1.0f,
                                           BoostDesc::create(BoostDesc::LBGM) );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BINBOOST_64, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-boostdesc-binboost_64",
                                            (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                            BoostDesc::create(BoostDesc::BINBOOST_64) );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BINBOOST_128, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-boostdesc-binboost_128",
                                            (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                            BoostDesc::create(BoostDesc::BINBOOST_128) );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BINBOOST_256, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-boostdesc-binboost_256",
                                            (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
                                            BoostDesc::create(BoostDesc::BINBOOST_256) );
    test.safe_run();
}
#endif  // OPENCV_XFEATURES2D_HAS_BOOST_DATA

#ifdef OPENCV_ENABLE_NONFREE
TEST(Features2d_BruteForceDescriptorMatcher_knnMatch, regression)
{
    const int sz = 100;
    const int k = 3;

    Ptr<DescriptorExtractor> ext = SURF::create();
    ASSERT_TRUE(ext);

    Ptr<FeatureDetector> det = SURF::create();
    //"%YAML:1.0\nhessianThreshold: 8000.\noctaves: 3\noctaveLayers: 4\nupright: 0\n"
    ASSERT_TRUE(det);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    ASSERT_TRUE(matcher);

    Mat imgT(256, 256, CV_8U, Scalar(255));
    line(imgT, Point(20, sz/2), Point(sz-21, sz/2), Scalar(100), 2);
    line(imgT, Point(sz/2, 20), Point(sz/2, sz-21), Scalar(100), 2);
    vector<KeyPoint> kpT;
    kpT.push_back( KeyPoint(50, 50, 16, 0, 20000, 1, -1) );
    kpT.push_back( KeyPoint(42, 42, 16, 160, 10000, 1, -1) );
    Mat descT;
    ext->compute(imgT, kpT, descT);

    Mat imgQ(256, 256, CV_8U, Scalar(255));
    line(imgQ, Point(30, sz/2), Point(sz-31, sz/2), Scalar(100), 3);
    line(imgQ, Point(sz/2, 30), Point(sz/2, sz-31), Scalar(100), 3);
    vector<KeyPoint> kpQ;
    det->detect(imgQ, kpQ);
    Mat descQ;
    ext->compute(imgQ, kpQ, descQ);

    vector<vector<DMatch> > matches;

    matcher->knnMatch(descQ, descT, matches, k);

    //cout << "\nBest " << k << " matches to " << descT.rows << " train desc-s." << endl;
    ASSERT_EQ(descQ.rows, static_cast<int>(matches.size()));
    for(size_t i = 0; i<matches.size(); i++)
    {
        //cout << "\nmatches[" << i << "].size()==" << matches[i].size() << endl;
        ASSERT_GE(min(k, descT.rows), static_cast<int>(matches[i].size()));
        for(size_t j = 0; j<matches[i].size(); j++)
        {
            //cout << "\t" << matches[i][j].queryIdx << " -> " << matches[i][j].trainIdx << endl;
            ASSERT_EQ(matches[i][j].queryIdx, static_cast<int>(i));
        }
    }
}
#endif

class CV_DetectPlanarTest : public cvtest::BaseTest
{
public:
    CV_DetectPlanarTest(const string& _fname, int _min_ninliers, const Ptr<Feature2D>& _f2d)
    : fname(_fname), min_ninliers(_min_ninliers), f2d(_f2d) {}

protected:
    void run(int)
    {
        if(f2d.empty())
            return;
        string path = string(ts->get_data_path()) + "detectors_descriptors_evaluation/planar/";
        string imgname1 = path + "box.png";
        string imgname2 = path + "box_in_scene.png";
        Mat img1 = imread(imgname1, 0);
        Mat img2 = imread(imgname2, 0);
        if( img1.empty() || img2.empty() )
        {
            ts->printf( cvtest::TS::LOG, "missing %s and/or %s\n", imgname1.c_str(), imgname2.c_str());
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }
        vector<KeyPoint> kpt1, kpt2;
        Mat d1, d2;
#ifdef HAVE_OPENCL
        if (cv::ocl::useOpenCL())
        {
            cv::UMat uimg1;
            img1.copyTo(uimg1);
            f2d->detectAndCompute(uimg1, Mat(), kpt1, d1);
            f2d->detectAndCompute(uimg1, Mat(), kpt2, d2);
        }
        else
#endif
        {
            f2d->detectAndCompute(img1, Mat(), kpt1, d1);
            f2d->detectAndCompute(img1, Mat(), kpt2, d2);
        }
        for( size_t i = 0; i < kpt1.size(); i++ )
            CV_Assert(kpt1[i].response > 0 );
        for( size_t i = 0; i < kpt2.size(); i++ )
            CV_Assert(kpt2[i].response > 0 );

        vector<DMatch> matches;
        BFMatcher(f2d->defaultNorm(), true).match(d1, d2, matches);

        vector<Point2f> pt1, pt2;
        for( size_t i = 0; i < matches.size(); i++ ) {
            pt1.push_back(kpt1[matches[i].queryIdx].pt);
            pt2.push_back(kpt2[matches[i].trainIdx].pt);
        }

        Mat inliers, H = findHomography(pt1, pt2, RANSAC, 10, inliers);
        int ninliers = countNonZero(inliers);

        if( ninliers < min_ninliers )
        {
            ts->printf( cvtest::TS::LOG, "too little inliers (%d) vs expected %d\n", ninliers, min_ninliers);
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }
    }

    string fname;
    int min_ninliers;
    Ptr<Feature2D> f2d;
};

TEST(Features2d_SIFTHomographyTest, regression) { CV_DetectPlanarTest test("SIFT", 80, SIFT::create()); test.safe_run(); }

#ifdef OPENCV_ENABLE_NONFREE
TEST(Features2d_SURFHomographyTest, regression) { CV_DetectPlanarTest test("SURF", 80, SURF::create()); test.safe_run(); }
#endif

class FeatureDetectorUsingMaskTest : public cvtest::BaseTest
{
public:
    FeatureDetectorUsingMaskTest(const Ptr<FeatureDetector>& featureDetector) :
        featureDetector_(featureDetector)
    {
        CV_Assert(featureDetector_);
    }

protected:

    void run(int)
    {
        const int nStepX = 2;
        const int nStepY = 2;

        const string imageFilename = string(ts->get_data_path()) + "/features2d/tsukuba.png";

        Mat image = imread(imageFilename);
        if(image.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imageFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        Mat mask(image.size(), CV_8U);

        const int stepX = image.size().width / nStepX;
        const int stepY = image.size().height / nStepY;

        vector<KeyPoint> keyPoints;
        vector<Point2f> points;
        for(int i=0; i<nStepX; ++i)
            for(int j=0; j<nStepY; ++j)
            {

                mask.setTo(0);
                Rect whiteArea(i * stepX, j * stepY, stepX, stepY);
                mask(whiteArea).setTo(255);

                featureDetector_->detect(image, keyPoints, mask);
                KeyPoint::convert(keyPoints, points);

                for(size_t k=0; k<points.size(); ++k)
                {
                    if ( !whiteArea.contains(points[k]) )
                    {
                        ts->printf(cvtest::TS::LOG, "The feature point is outside of the mask.");
                        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                        return;
                    }
                }
            }

        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector_;
};

TEST(Features2d_SIFT_using_mask, regression)
{
    FeatureDetectorUsingMaskTest test(SIFT::create());
    test.safe_run();
}

#ifdef OPENCV_ENABLE_NONFREE
TEST(DISABLED_Features2d_SURF_using_mask, regression)
{
    FeatureDetectorUsingMaskTest test(SURF::create());
    test.safe_run();
}
#endif // NONFREE

}} // namespace
