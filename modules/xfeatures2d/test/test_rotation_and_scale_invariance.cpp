// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

#include "features2d/test/test_detectors_invariance.impl.hpp" // main OpenCV repo
#include "features2d/test/test_descriptors_invariance.impl.hpp" // main OpenCV repo

namespace opencv_test { namespace {

static const char* const IMAGE_TSUKUBA = "features2d/tsukuba.png";
static const char* const IMAGE_BIKES = "detectors_descriptors_evaluation/images_datasets/bikes/img1.png";

// ========================== ROTATION INVARIANCE =============================

#ifdef OPENCV_ENABLE_NONFREE

INSTANTIATE_TEST_CASE_P(SURF, DetectorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA, SURF::create(), 0.40f, 0.76f)
));

INSTANTIATE_TEST_CASE_P(SURF, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA, SURF::create(), SURF::create(), 0.83f)
));

#endif // NONFREE

INSTANTIATE_TEST_CASE_P(LATCH, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA, SIFT::create(), LATCH::create(), 0.98f)
));

INSTANTIATE_TEST_CASE_P(BEBLID, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA, SIFT::create(), BEBLID::create(6.75), 0.98f)
));

INSTANTIATE_TEST_CASE_P(DAISY, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               BRISK::create(),
               DAISY::create(15, 3, 8, 8, DAISY::NRM_NONE, noArray(), true, true),
               0.79f)
));
#ifdef OPENCV_XFEATURES2D_HAS_VGG_DATA
INSTANTIATE_TEST_CASE_P(VGG120, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               KAZE::create(),
               VGG::create(VGG::VGG_120, 1.4f, true, true, 48.0f, false),
               0.97f)
));
INSTANTIATE_TEST_CASE_P(VGG80, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               KAZE::create(),
               VGG::create(VGG::VGG_80, 1.4f, true, true, 48.0f, false),
               0.97f)
));
INSTANTIATE_TEST_CASE_P(VGG64, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               KAZE::create(),
               VGG::create(VGG::VGG_64, 1.4f, true, true, 48.0f, false),
               0.97f)
));
INSTANTIATE_TEST_CASE_P(VGG48, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               KAZE::create(),
               VGG::create(VGG::VGG_48, 1.4f, true, true, 48.0f, false),
               0.97f)
));
#endif  // OPENCV_XFEATURES2D_HAS_VGG_DATA

#ifdef OPENCV_ENABLE_NONFREE

INSTANTIATE_TEST_CASE_P(BRIEF_64, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BriefDescriptorExtractor::create(64,true),
               0.98f)
));

INSTANTIATE_TEST_CASE_P(BRIEF_32, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BriefDescriptorExtractor::create(32,true),
               0.97f)
));

INSTANTIATE_TEST_CASE_P(BRIEF_16, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BriefDescriptorExtractor::create(16, true),
               0.98f)
));

INSTANTIATE_TEST_CASE_P(FREAK, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               FREAK::create(),
               0.90f)
));

#ifdef OPENCV_XFEATURES2D_HAS_BOOST_DATA
INSTANTIATE_TEST_CASE_P(BoostDesc_BGM, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::BGM, true, 6.25f),
               0.999f)
));

INSTANTIATE_TEST_CASE_P(BoostDesc_BGM_HARD, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::BGM_HARD, true, 6.25f),
               0.98f)
));

INSTANTIATE_TEST_CASE_P(BoostDesc_BGM_BILINEAR, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::BGM_BILINEAR, true, 6.25f),
               0.98f)
));

INSTANTIATE_TEST_CASE_P(BoostDesc_BGM_LBGM, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::LBGM, true, 6.25f),
               0.999f)
));

INSTANTIATE_TEST_CASE_P(BoostDesc_BINBOOST_64, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::BINBOOST_64, true, 6.25f),
               0.98f)
));

INSTANTIATE_TEST_CASE_P(BoostDesc_BINBOOST_128, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::BINBOOST_128, true, 6.25f),
               0.98f)
));

INSTANTIATE_TEST_CASE_P(BoostDesc_BINBOOST_256, DescriptorRotationInvariance, Values(
    make_tuple(IMAGE_TSUKUBA,
               SURF::create(),
               BoostDesc::create(BoostDesc::BINBOOST_256, true, 6.25f),
               0.999f)
));
#endif  // OPENCV_XFEATURES2D_HAS_BOOST_DATA
#endif



// ============================ SCALE INVARIANCE ==============================

#ifdef OPENCV_ENABLE_NONFREE
INSTANTIATE_TEST_CASE_P(SURF, DetectorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES, SURF::create(), 0.64f, 0.84f)
));

INSTANTIATE_TEST_CASE_P(SURF, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES, SURF::create(), SURF::create(), 0.7f)
));
#endif // NONFREE


#if 0  // DAISY is not scale invariant
INSTANTIATE_TEST_CASE_P(DISABLED_DAISY, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               BRISK::create(),
               DAISY::create(15, 3, 8, 8, DAISY::NRM_NONE, noArray(), true, true),
               0.1f)
));
#endif

#ifdef OPENCV_XFEATURES2D_HAS_VGG_DATA
INSTANTIATE_TEST_CASE_P(VGG120, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               KAZE::create(),
               VGG::create(VGG::VGG_120, 1.4f, true, true, 48.0f, false),
               0.98f)
));
INSTANTIATE_TEST_CASE_P(VGG80, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               KAZE::create(),
               VGG::create(VGG::VGG_80, 1.4f, true, true, 48.0f, false),
               0.98f)
));
INSTANTIATE_TEST_CASE_P(VGG64, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               KAZE::create(),
               VGG::create(VGG::VGG_64, 1.4f, true, true, 48.0f, false),
               0.97f)
));
INSTANTIATE_TEST_CASE_P(VGG48, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               KAZE::create(),
               VGG::create(VGG::VGG_48, 1.4f, true, true, 48.0f, false),
               0.93f)
));
#endif  // OPENCV_XFEATURES2D_HAS_VGG_DATA

#ifdef OPENCV_ENABLE_NONFREE  // SURF detector is used in tests
#ifdef OPENCV_XFEATURES2D_HAS_BOOST_DATA
INSTANTIATE_TEST_CASE_P(BoostDesc_BGM, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::BGM, true, 6.25f),
               0.98f)
));
INSTANTIATE_TEST_CASE_P(BoostDesc_BGM_HARD, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::BGM_HARD, true, 6.25f),
               0.75f)
));
INSTANTIATE_TEST_CASE_P(BoostDesc_BGM_BILINEAR, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::BGM_BILINEAR, true, 6.25f),
               0.95f)
));
INSTANTIATE_TEST_CASE_P(BoostDesc_LBGM, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::LBGM, true, 6.25f),
               0.95f)
));
INSTANTIATE_TEST_CASE_P(BoostDesc_BINBOOST_64, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::BINBOOST_64, true, 6.25f),
               0.75f)
));
INSTANTIATE_TEST_CASE_P(BoostDesc_BINBOOST_128, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::BINBOOST_128, true, 6.25f),
               0.95f)
));
INSTANTIATE_TEST_CASE_P(BoostDesc_BINBOOST_256, DescriptorScaleInvariance, Values(
    make_tuple(IMAGE_BIKES,
               SURF::create(),
               BoostDesc::create(BoostDesc::BINBOOST_256, true, 6.25f),
               0.98f)
));
#endif  // OPENCV_XFEATURES2D_HAS_BOOST_DATA
#endif // NONFREE



// ============================== OTHER TESTS =================================

#ifdef OPENCV_ENABLE_NONFREE
TEST(Features2d_RotationInvariance2_Detector_SURF, regression)
{
    Mat cross(100, 100, CV_8UC1, Scalar(255));
    line(cross, Point(30, 50), Point(69, 50), Scalar(100), 3);
    line(cross, Point(50, 30), Point(50, 69), Scalar(100), 3);

    Ptr<SURF> surf = SURF::create(8000., 3, 4, true, false);

    vector<KeyPoint> keypoints;
    surf->detect(cross, keypoints);

    // Expect 5 keypoints.  One keypoint has coordinates (50.0, 50.0).
    // The other 4 keypoints should have the same response.
    // The order of the keypoints is indeterminate.
    ASSERT_EQ(keypoints.size(), (vector<KeyPoint>::size_type) 5);

    int i1 = -1;
    for(int i = 0; i < 5; i++)
    {
        if(keypoints[i].pt.x == 50.0f)
            ;
        else if(i1 == -1)
            i1 = i;
        else
            ASSERT_LT(fabs(keypoints[i1].response - keypoints[i].response) / keypoints[i1].response, 1e-6);
    }
}

#endif // NONFREE

}} // namespace
