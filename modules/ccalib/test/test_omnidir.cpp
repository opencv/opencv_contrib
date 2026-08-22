// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Regression coverage for opencv/opencv#28462: cv::omnidir::calibrate() /
// cv::omnidir::stereoCalibrate() used to fail with a misleading
// "objectPoints.type() == CV_64FC3" assertion when the filtered point
// vectors ended up empty after initial-pose-estimation view filtering.
// The real cause is emptiness, not a type mismatch. These tests verify the
// clearer, actionable error introduced to replace that misleading message.

static Mat randomPoints3f(int n, RNG& rng)
{
    Mat pts(n, 1, CV_32FC3);
    for (int i = 0; i < n; ++i)
        pts.at<Vec3f>(i) = Vec3f(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f));
    return pts;
}

static Mat randomPoints2f(int n, RNG& rng)
{
    Mat pts(n, 1, CV_32FC2);
    for (int i = 0; i < n; ++i)
        pts.at<Vec2f>(i) = Vec2f(rng.uniform(0.f, 640.f), rng.uniform(0.f, 480.f));
    return pts;
}

// Builds a small planar grid of object points and their exact projections
// under the given camera model / pose, so the pair is self-consistent and
// initializeCalibration()'s closed-form pose estimate should succeed.
static void generateValidView(const Matx33d& K, double xi, const Matx14d& D,
    const Vec3d& rvec, const Vec3d& tvec, Mat& objPoints, Mat& imgPoints)
{
    const int gridSize = 6; // 36 points, well above the minimum needed for the linear pose solve
    objPoints = Mat(gridSize * gridSize, 1, CV_32FC3);
    int idx = 0;
    for (int r = 0; r < gridSize; ++r)
        for (int c = 0; c < gridSize; ++c)
            objPoints.at<Vec3f>(idx++) = Vec3f(0.1f * c, 0.1f * r, 0.f);

    Mat imgPointsD;
    cv::omnidir::projectPoints(objPoints, imgPointsD, rvec, tvec, K, xi, D, cv::noArray());
    imgPointsD.convertTo(imgPoints, CV_32FC2);
}

TEST(CV_OmnidirCalibrate, throws_clear_error_when_no_views_survive_initialization)
{
    RNG& rng = cv::theRNG();
    std::vector<Mat> objectPoints, imagePoints;
    const int nViews = 5;
    const int nPointsPerView = 30;
    for (int v = 0; v < nViews; ++v)
    {
        // Fully independent random object/image points: not a real calibration
        // target, so every view is expected to fail initial pose estimation.
        objectPoints.push_back(randomPoints3f(nPointsPerView, rng));
        imagePoints.push_back(randomPoints2f(nPointsPerView, rng));
    }

    Mat K, xi, D;
    std::vector<Mat> rvecs, tvecs;
    bool threw = false;
    std::string message;
    try
    {
        cv::omnidir::calibrate(objectPoints, imagePoints, Size(640, 480), K, xi, D,
            rvecs, tvecs, 0, TermCriteria(3, 100, 1e-6), cv::noArray());
    }
    catch (const cv::Exception& e)
    {
        threw = true;
        message = e.what();
    }

    ASSERT_TRUE(threw) << "calibrate() should throw when no views survive initialization";
    EXPECT_NE(message.find("no calibration views survived"), std::string::npos)
        << "exception message should explain the actual cause, got: " << message;
    EXPECT_EQ(message.find("CV_64FC3"), std::string::npos)
        << "exception message should not resurface the misleading type-assertion text, got: " << message;
}

TEST(CV_OmnidirCalibrate, succeeds_with_a_valid_synthetic_view)
{
    Matx33d K(400, 0, 320, 0, 400, 240, 0, 0, 1);
    double xi = 1.0;
    Matx14d D(0, 0, 0, 0);

    std::vector<Mat> objectPoints, imagePoints;
    Mat objPts, imgPts;
    generateValidView(K, xi, D, Vec3d(0.05, -0.02, 0.01), Vec3d(0, 0, 3), objPts, imgPts);
    objectPoints.push_back(objPts);
    imagePoints.push_back(imgPts);
    // A second, slightly different valid view so calibration has more than one sample.
    generateValidView(K, xi, D, Vec3d(-0.03, 0.04, 0.0), Vec3d(0.1, -0.05, 3.2), objPts, imgPts);
    objectPoints.push_back(objPts);
    imagePoints.push_back(imgPts);

    Mat Kout, xiOut, Dout;
    std::vector<Mat> rvecs, tvecs;
    Mat idx;
    ASSERT_NO_THROW(
        cv::omnidir::calibrate(objectPoints, imagePoints, Size(640, 480), Kout, xiOut, Dout,
            rvecs, tvecs, 0, TermCriteria(3, 100, 1e-6), idx)
    ) << "calibrate() should not throw the new guard error for a valid, self-consistent view";
    EXPECT_GT((int)idx.total(), 0) << "at least one synthetic view should survive initialization";
}

TEST(CV_OmnidirStereoCalibrate, throws_clear_error_when_camera_valid_view_sets_do_not_intersect)
{
    Matx33d K1(400, 0, 320, 0, 400, 240, 0, 0, 1);
    Matx33d K2(410, 0, 330, 0, 410, 250, 0, 0, 1);
    double xi1 = 1.0, xi2 = 1.0;
    Matx14d D1(0, 0, 0, 0), D2(0, 0, 0, 0);
    RNG& rng = cv::theRNG();

    std::vector<Mat> objectPoints, imagePoints1, imagePoints2;

    // Views 0,1: camera 1 sees a valid, self-consistent pattern; camera 2 sees
    // unrelated random points for the same nominal views.
    for (int v = 0; v < 2; ++v)
    {
        Mat objPts, img1;
        generateValidView(K1, xi1, D1, Vec3d(0.02 * v, -0.01, 0.0), Vec3d(0.0, 0.0, 3.0 + 0.1 * v), objPts, img1);
        objectPoints.push_back(objPts);
        imagePoints1.push_back(img1);
        imagePoints2.push_back(randomPoints2f(objPts.rows, rng));
    }

    // Views 2,3: camera 2 sees a valid, self-consistent pattern (reusing the
    // same object points as their matching view); camera 1 sees unrelated
    // random points instead.
    for (int v = 0; v < 2; ++v)
    {
        Mat objPts, img2;
        generateValidView(K2, xi2, D2, Vec3d(-0.02 * v, 0.03, 0.0), Vec3d(0.05, 0.0, 3.2 + 0.1 * v), objPts, img2);
        objectPoints.push_back(objPts);
        imagePoints2.push_back(img2);
        imagePoints1.push_back(randomPoints2f(objPts.rows, rng));
    }

    Mat Kout1, xiOut1, Dout1, Kout2, xiOut2, Dout2, rvec, tvec;
    std::vector<Mat> rvecsL, tvecsL;
    bool threw = false;
    std::string message;
    try
    {
        cv::omnidir::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, Size(640, 480), Size(640, 480),
            Kout1, xiOut1, Dout1, Kout2, xiOut2, Dout2, rvec, tvec, rvecsL, tvecsL,
            0, TermCriteria(3, 100, 1e-6), cv::noArray());
    }
    catch (const cv::Exception& e)
    {
        threw = true;
        message = e.what();
    }

    ASSERT_TRUE(threw) << "stereoCalibrate() should throw when the per-camera valid-view sets do not intersect";
    EXPECT_NE(message.find("no calibration views are valid for both cameras"), std::string::npos)
        << "exception message should name the intersection-specific cause, got: " << message;
    EXPECT_EQ(message.find("CV_64FC3"), std::string::npos)
        << "exception message should not resurface the misleading type-assertion text, got: " << message;
}

}} // namespace
