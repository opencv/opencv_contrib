// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_ridgedetectionfilter, ReferenceAccuracy)
{
    String openCVExtraDir = cvtest::TS::ptr()->get_data_path();
    String srcImgPath = "cv/ximgproc/sources/04.png";
    String refPath = "cv/ximgproc/results/ridge_filter_test_ref/04.png";
    Mat src = imread(openCVExtraDir + srcImgPath);
    Mat ref = imread(openCVExtraDir + refPath, 0);
    Mat n_ref;
    ref.convertTo(n_ref, CV_8UC1);
    Ptr<RidgeDetectionFilter> rdf = RidgeDetectionFilter::create();
    Mat out;
    rdf->getRidgeFilteredImage(src, out);
    Mat out_cmp;
    out.convertTo(out_cmp, CV_8UC1);
    EXPECT_LE(cvtest::norm(out, ref, NORM_INF), 0.0f);
    EXPECT_LE(cvtest::norm(out, ref, NORM_L2 | NORM_RELATIVE), .0f);
}

TEST(ximgproc_ridgedetectionfilter, Synthetic32FAnd64FExactFormulaParity)
{
    RNG& rng = theRNG();
    Mat src(64, 80, CV_8UC3);
    rng.fill(src, RNG::UNIFORM, 0, 256);

    for (int ddepth : { CV_32FC1, CV_64FC1 })
    {
        Ptr<RidgeDetectionFilter> rdf = RidgeDetectionFilter::create(ddepth, 1, 1, 3, CV_32FC1, 1.0, 0.0, BORDER_DEFAULT);
        Mat out;
        rdf->getRidgeFilteredImage(src, out);

        Mat gray, sbx, sby, sbxx, sbyy, sbxy;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Sobel(gray, sbx, ddepth, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
        Sobel(gray, sby, ddepth, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT);
        Sobel(sbx, sbxx, ddepth, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
        Sobel(sby, sbyy, ddepth, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT);
        Sobel(sbx, sbxy, ddepth, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT);

        Mat sb2xx, sb2yy, sb2xy, sbxxyy, rootex, root, ref;
        cv::multiply(sbxx, sbxx, sb2xx);
        cv::multiply(sbyy, sbyy, sb2yy);
        cv::multiply(sbxy, sbxy, sb2xy);
        cv::multiply(sbxx, sbyy, sbxxyy);
        rootex = (sb2xx + (sb2xy + sb2xy + sb2xy + sb2xy) - (sbxxyy + sbxxyy) + sb2yy);
        sqrt(rootex, root);
        Mat ridgexp = ((sbxx + sbyy) + root);
        ridgexp.convertTo(ref, CV_32FC1, 0.5);

        EXPECT_LE(cvtest::norm(out, ref, NORM_INF), 1e-5);
    }
}

}} // namespace
