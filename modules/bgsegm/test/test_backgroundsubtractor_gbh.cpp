// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: andrewgodbehere
#include "test_precomp.hpp"

namespace opencv_test { namespace {

/**
 * This test checks the following:
 * (i) BackgroundSubtractorGMG can operate with matrices of various types and sizes
 * (ii) Training mode returns empty fgmask
 * (iii) End of training mode, and anomalous frame yields every pixel detected as FG
 */
typedef testing::TestWithParam<std::tuple<perf::MatDepth,int>> bgsubgmg_allTypes;
TEST_P(bgsubgmg_allTypes, accuracy)
{
    const int depth = get<0>(GetParam());
    const int ncn   = get<1>(GetParam());
    const int mtype = CV_MAKETYPE(depth, ncn);
    const int width  = 64;
    const int height = 64;
    RNG& rng = TS::ptr()->get_rng();

    Ptr<BackgroundSubtractorGMG> fgbg = createBackgroundSubtractorGMG();
    ASSERT_TRUE(fgbg != nullptr) << "Failed to call createBackgroundSubtractorGMG()";

    /**
     * Set a few parameters
     */
    fgbg->setSmoothingRadius(7);
    fgbg->setDecisionThreshold(0.7);
    fgbg->setNumFrames(120);

    /**
     * Generate bounds for the values in the matrix for each type
     */
    double maxd = 0, mind = 0;

    /**
     * Max value for simulated images picked randomly in upper half of type range
     * Min value for simulated images picked randomly in lower half of type range
     */
    if (depth == CV_8U)
    {
        uchar half = UCHAR_MAX/2;
        maxd = (unsigned char)rng.uniform(half+32, UCHAR_MAX);
        mind = (unsigned char)rng.uniform(0, half-32);
    }
    else if (depth == CV_8S)
    {
        maxd = (char)rng.uniform(32, CHAR_MAX);
        mind = (char)rng.uniform(CHAR_MIN, -32);
    }
    else if (depth == CV_16U)
    {
        ushort half = USHRT_MAX/2;
        maxd = (unsigned int)rng.uniform(half+32, USHRT_MAX);
        mind = (unsigned int)rng.uniform(0, half-32);
    }
    else if (depth == CV_16S)
    {
        maxd = rng.uniform(32, SHRT_MAX);
        mind = rng.uniform(SHRT_MIN, -32);
    }
    else if (depth == CV_32S)
    {
        maxd = rng.uniform(32, INT_MAX);
        mind = rng.uniform(INT_MIN, -32);
    }
    else
    {
        ASSERT_TRUE( (depth == CV_32F)||(depth == CV_64F) ) << "Unsupported depth";
        const double harf = 0.5;
        const double bias = 0.125; // = 32/256 (Like CV_8U)
        maxd = rng.uniform(harf + bias, 1.0);
        mind = rng.uniform(0.0, harf - bias );
    }

    fgbg->setMinVal(mind);
    fgbg->setMaxVal(maxd);

    Mat simImage(height, width, mtype);
    Mat fgmask;

    const Mat fullbg(height, width, CV_8UC1, cv::Scalar(0)); // all background.

    const int numLearningFrames = 120;
    for (int i = 0; i < numLearningFrames; ++i)
    {
        /**
         * Genrate simulated "image" for any type. Values always confined to upper half of range.
         */
        rng.fill(simImage, RNG::UNIFORM, (mind + maxd)*0.5, maxd);

        /**
         * Feed simulated images into background subtractor
         */
        fgbg->apply(simImage,fgmask);

        EXPECT_EQ(cv::norm(fgmask, fullbg, NORM_INF), 0) << "foreground mask should be entirely background during training";
    }
    //! generate last image, distinct from training images
    rng.fill(simImage, RNG::UNIFORM, mind, maxd);
    fgbg->apply(simImage,fgmask);

    const Mat fullfg(height, width, CV_8UC1, cv::Scalar(255)); // all foreground.
    EXPECT_EQ(cv::norm(fgmask, fullfg, NORM_INF), 0) << "foreground mask should be entirely foreground finally";
}

INSTANTIATE_TEST_CASE_P(/**/,
                        bgsubgmg_allTypes,
                        testing::Combine(
                            testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            testing::Values(1,2,3,4)));

}} // namespace
