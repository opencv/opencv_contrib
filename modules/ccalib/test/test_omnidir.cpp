// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

template <typename T>
static void runOmnidirNonContiguousTest(int depth)
{
    typedef Vec<T, 3> Vec3T;
    typedef Vec<T, 2> Vec2T;

    const int N = 12;
    const int pt3Type = CV_MAKETYPE(depth, 3);
    const int pt2Type = CV_MAKETYPE(depth, 2);

    Mat K = (Mat_<double>(3, 3) << 320.0, 0.5, 319.5,
                                     0.0, 321.0, 239.5,
                                     0.0,   0.0,   1.0);
    K.convertTo(K, depth);

    // Store rvec, tvec, D as columns inside larger matrices so .col(...) is non-contiguous
    Mat poseCols(3, 3, depth, Scalar::all(99));
    Mat rvecCont = (Mat_<double>(3, 1) << 0.08, -0.12, 0.05);
    Mat tvecCont = (Mat_<double>(3, 1) << 0.15, -0.05, 1.80);
    rvecCont.convertTo(poseCols.col(0), depth);
    tvecCont.convertTo(poseCols.col(2), depth);
    Mat rvecSlice = poseCols.col(0);
    Mat tvecSlice = poseCols.col(2);
    ASSERT_FALSE(rvecSlice.isContinuous());
    ASSERT_FALSE(tvecSlice.isContinuous());

    Mat distCols(4, 2, depth, Scalar::all(-77));
    Mat DCont = (Mat_<double>(4, 1) << -0.18, 0.04, 0.001, -0.002);
    DCont.convertTo(distCols.col(1), depth);
    Mat DSlice = distCols.col(1);
    ASSERT_FALSE(DSlice.isContinuous());

    const double xiVal = 0.85;
    Mat xiCols(2, 2, depth, Scalar::all(42));
    xiCols.at<T>(1, 1) = static_cast<T>(xiVal);
    Mat xiSlice = xiCols(Rect(1, 1, 1, 1));

    // Create N x 1 3-channel object points inside a wider N x 3 matrix (simulating NumPy slice view)
    Mat objCanvas(N, 3, pt3Type, Scalar::all(12345));
    for (int i = 0; i < N; ++i)
    {
        double angle = 0.5 * i;
        objCanvas.at<Vec3T>(i, 1) = Vec3T(static_cast<T>(0.3 * std::cos(angle)),
                                          static_cast<T>(0.3 * std::sin(angle)),
                                          static_cast<T>(1.0 + 0.1 * i));
    }
    Mat objSlice = objCanvas.col(1);
    Mat objCont = objSlice.clone();
    ASSERT_FALSE(objSlice.isContinuous());
    ASSERT_TRUE(objCont.isContinuous());

    // 1. projectPoints: non-contiguous inputs + non-contiguous pre-allocated outputs + jacobian
    Mat imgExpected, jacExpected;
    omnidir::projectPoints(objCont, imgExpected, rvecSlice.clone(), tvecSlice.clone(), K, xiVal, DSlice.clone(), jacExpected);

    Mat imgOutCanvas(N + 4, 3, pt2Type, Scalar::all(-999));
    Mat imgOutSlice = imgOutCanvas(Rect(1, 2, 1, N));
    Mat jacOutCanvas(2 * N + 4, 20, CV_64F, Scalar::all(-999));
    Mat jacOutSlice = jacOutCanvas(Rect(2, 1, 16, 2 * N));
    ASSERT_FALSE(imgOutSlice.isContinuous());
    ASSERT_FALSE(jacOutSlice.isContinuous());

    omnidir::projectPoints(objSlice, imgOutSlice, rvecSlice, tvecSlice, K, xiVal, DSlice, jacOutSlice);
    EXPECT_EQ(0.0, cv::norm(imgExpected, imgOutSlice, NORM_INF));
    EXPECT_EQ(0.0, cv::norm(jacExpected, jacOutSlice, NORM_INF));
    EXPECT_EQ(Vec2T(static_cast<T>(-999), static_cast<T>(-999)), imgOutCanvas.at<Vec2T>(2, 0));
    EXPECT_EQ(Vec2T(static_cast<T>(-999), static_cast<T>(-999)), imgOutCanvas.at<Vec2T>(2, 2));
    EXPECT_EQ(-999.0, jacOutCanvas.at<double>(1, 1));
    EXPECT_EQ(-999.0, jacOutCanvas.at<double>(1, 18));

    // Also test fresh (unallocated) OutputArray with non-contiguous inputs
    Mat imgFresh;
    omnidir::projectPoints(objSlice, imgFresh, rvecSlice, tvecSlice, K, xiVal, DSlice);
    EXPECT_EQ(0.0, cv::norm(imgExpected, imgFresh, NORM_INF));

    // 2. undistortPoints: non-contiguous distorted points (issue #3625 reproducer) + non-contiguous D/R/output + 1x3 row rvec
    Mat distPtsCanvas(N, 3, pt2Type, Scalar::all(54321));
    imgExpected.copyTo(distPtsCanvas.col(1));
    Mat distPtsSlice = distPtsCanvas.col(1);
    Mat distPtsCont = distPtsSlice.clone();
    ASSERT_FALSE(distPtsSlice.isContinuous());

    Mat undistExpected;
    omnidir::undistortPoints(distPtsCont, undistExpected, K, DSlice.clone(), xiSlice.clone(), rvecSlice.clone());

    Mat rvecRow = rvecSlice.clone().t(); // 1x3 row vector
    Mat undistFresh;
    omnidir::undistortPoints(distPtsSlice, undistFresh, K, DSlice, xiSlice, rvecRow);
    EXPECT_EQ(0.0, cv::norm(undistExpected, undistFresh, NORM_INF));

    Mat undistOutCanvas(N + 2, 3, pt2Type, Scalar::all(-888));
    Mat undistOutSlice = undistOutCanvas(Rect(2, 1, 1, N));
    ASSERT_FALSE(undistOutSlice.isContinuous());
    omnidir::undistortPoints(distPtsSlice, undistOutSlice, K, DSlice, xiSlice, rvecSlice);
    EXPECT_EQ(0.0, cv::norm(undistExpected, undistOutSlice, NORM_INF));
    EXPECT_EQ(Vec2T(static_cast<T>(-888), static_cast<T>(-888)), undistOutCanvas.at<Vec2T>(1, 1));
}

TEST(CCalib_Omnidir, non_contiguous_64f)
{
    runOmnidirNonContiguousTest<double>(CV_64F);
}

TEST(CCalib_Omnidir, non_contiguous_32f)
{
    runOmnidirNonContiguousTest<float>(CV_32F);
}

TEST(CCalib_Omnidir, empty_points)
{
    Mat K = Mat::eye(3, 3, CV_64F);
    Mat D = Mat::zeros(4, 1, CV_64F);
    Mat rvec = Mat::zeros(3, 1, CV_64F);
    Mat tvec = (Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    Mat xi = (Mat_<double>(1, 1) << 0.5);

    Mat emptyObj(0, 1, CV_64FC3), imgPts, jac;
    omnidir::projectPoints(emptyObj, imgPts, rvec, tvec, K, 0.5, D, jac);
    EXPECT_TRUE(imgPts.empty());
    EXPECT_TRUE(jac.empty());

    Mat emptyDist(0, 1, CV_64FC2), undistPts;
    omnidir::undistortPoints(emptyDist, undistPts, K, D, xi, rvec);
    EXPECT_TRUE(undistPts.empty());
}

}} // namespace
