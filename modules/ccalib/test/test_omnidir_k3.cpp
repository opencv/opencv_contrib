
/**
 * @file test_omnidir_k3_hard.cpp
 *
 * Hard tests for omnidir 5-parameter distortion (k3).
 *
 * Goals:
 * 1) Strong checks:
 *    - Real-data regression with "golden" calibration parameters (K, xi, D4).
 *    - Deterministic synthetic test with known K/xi/D (including k3 != 0),
 *      verifying coefficient recovery and reprojection RMS.
 *
 * 2) Backward compatibility:
 *    - 4 coeffs == 5 coeffs with k3=0 (projectPoints / undistort map consistency).
 *
 * Data:
 *   Uses real chessboard images from opencv_extra:
 *     <data_path>/cv/stereo/case1/left*.png
 */

#include "test_precomp.hpp"

#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib> 


namespace opencv_test {

using namespace cv;
using std::vector;

// =====================================================
// Helpers
// =====================================================

static double maxAbsDiff(const Mat& a, const Mat& b)
{
    CV_Assert(a.size() == b.size());
    CV_Assert(a.type() == b.type());
    Mat diff;
    absdiff(a, b, diff);
    double maxv = 0.0;
    minMaxLoc(diff.reshape(1), nullptr, &maxv);
    return maxv;
}

static vector<Point3f> makeChessboard3D(Size board, float square)
{
    vector<Point3f> obj;
    obj.reserve((size_t)board.area());
    for (int y = 0; y < board.height; ++y)
        for (int x = 0; x < board.width; ++x)
            obj.emplace_back((float)x * square, (float)y * square, 0.f);
    return obj;
}

static bool detectCorners(const Mat& gray, Size board, vector<Point2f>& corners)
{
    bool ok = findChessboardCorners(gray, board, corners,
        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

    if (!ok)
        return false;

    cornerSubPix(gray, corners,
        Size(11,11), Size(-1,-1),
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 1e-4));

    return true;
}

/**
 * Compute reprojection RMS using omnidir::projectPoints.
 * If idx is non-empty, only the used views are evaluated (calibrate may reject some).
 */
static double computeReprojRms(const vector<vector<Point3f>>& objectPoints,
                              const vector<vector<Point2f>>& imagePoints,
                              const Mat& K, double xi, const Mat& D,
                              const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                              const Mat& idx)
{
    CV_Assert(objectPoints.size() == imagePoints.size());
    CV_Assert(rvecs.size() == tvecs.size());

    const bool useIdx = !idx.empty();
    const size_t nUsed = useIdx ? (size_t)idx.total() : objectPoints.size();
    CV_Assert(rvecs.size() == nUsed);

    Mat idxRow;
    if (useIdx)
    {
        idxRow = idx.reshape(1, 1);
        if (idxRow.type() != CV_32S)
            idxRow.convertTo(idxRow, CV_32S);
    }

    double sse = 0.0;
    size_t n = 0;

    for (size_t i = 0; i < nUsed; ++i)
    {
        const int view = useIdx ? idxRow.at<int>((int)i) : (int)i;

        vector<Point2f> proj;
        cv::omnidir::projectPoints(objectPoints[(size_t)view], proj,
                                   rvecs[i], tvecs[i], K, xi, D);

        CV_Assert(proj.size() == imagePoints[(size_t)view].size());

        for (size_t j = 0; j < proj.size(); ++j)
        {
            const double dx = (double)proj[j].x - (double)imagePoints[(size_t)view][j].x;
            const double dy = (double)proj[j].y - (double)imagePoints[(size_t)view][j].y;
            sse += dx*dx + dy*dy;
        }
        n += proj.size();
    }

    return (n > 0) ? std::sqrt(sse / (double)n) : 0.0;
}

// =====================================================
// Real-data loader (opencv_extra)
// =====================================================

static void loadRealData(Size board, float square,
                         vector<vector<Point3f>>& objectPoints,
                         vector<vector<Point2f>>& imagePoints,
                         Size& imageSize)
{
    // 1) Prefer explicit OPENCV_TEST_DATA_PATH if available (most robust)
    const char* env = std::getenv("OPENCV_TEST_DATA_PATH");

    std::string root;
    if (env && *env)
    {
        root = std::string(env);
        if (!root.empty() && root.back() != '/')
            root.push_back('/');
    }
    else
    {
        // 2) Fallback: OpenCV test infra path (may point to .../testdata/ccalib/)
        root = cvtest::TS::ptr()->get_data_path();
        if (!root.empty() && root.back() != '/')
            root.push_back('/');
    }

    // Candidate directories to locate the dataset
    const std::vector<std::string> candidates = {
        root + "cv/stereo/case1/",
        root + "stereo/case1/",
        root + "cv/stereo/",
        root + "cv/"
    };

    std::string dir;
    vector<String> files;

    // Find first candidate that contains left*.png
    for (const auto& c : candidates)
    {
        if (!cv::utils::fs::exists(c))
            continue;

        files.clear();
        cv::glob(c + "left*.png", files, false);
        if (!files.empty())
        {
            dir = c;
            break;
        }
    }

    if (dir.empty())
    {
        throw cvtest::SkipTestException(
            "Could not locate stereo case1 images. "
            "Checked candidates under root=" + root +
            "\nExpected something like: cv/stereo/case1/left*.png");
    }

    if (files.size() < 5)
        throw cvtest::SkipTestException("Not enough images in: " + dir + " (need >= 5 left*.png)");

    objectPoints.clear();
    imagePoints.clear();
    imageSize = Size();

    const auto obj = makeChessboard3D(board, square);

    for (const auto& f : files)
    {
        Mat img = imread(f, IMREAD_GRAYSCALE);
        if (img.empty())
            continue;

        if (imageSize.empty())
            imageSize = img.size();

        vector<Point2f> corners;
        if (!detectCorners(img, board, corners))
            continue;

        objectPoints.push_back(obj);
        imagePoints.push_back(corners);
    }

    if (objectPoints.size() < 5)
        throw cvtest::SkipTestException("Not enough valid chessboard detections in: " + dir);
}

// =====================================================
// Synthetic data generator (deterministic, k3 observable)
// =====================================================

static void synthesizeOmniData(int nViews,
                               const vector<Point3f>& obj,
                               const Size& imageSize,
                               const Matx33d& Ktrue, double xiTrue,
                               const Mat& Dtrue,
                               vector<vector<Point3f>>& objectPoints,
                               vector<vector<Point2f>>& imagePoints,
                               double noiseSigmaPx = 0.25,
                               uint64 seed = 0xBEEF)
{
    RNG rng((uint64)seed);

    objectPoints.clear();
    imagePoints.clear();
    objectPoints.reserve((size_t)nViews);
    imagePoints.reserve((size_t)nViews);

    const double cx = Ktrue(0,2);
    const double cy = Ktrue(1,2);
    const int margin = 8; // pixels

    int tries = 0;
    const int maxTries = 20000;

    while ((int)imagePoints.size() < nViews && tries++ < maxTries)
    {
        // Make the board large in the image -> k3 becomes observable
        // Smaller z => larger projection radius (but keep it in front)
        Vec3d r(rng.uniform(-0.35, 0.35),
                rng.uniform(-0.35, 0.35),
                rng.uniform(-0.35, 0.35));

        Vec3d t(rng.uniform(-0.25, 0.25),
                rng.uniform(-0.25, 0.25),
                rng.uniform(0.55, 0.95));

        Mat rvec(r), tvec(t);

        vector<Point2f> img;
        cv::omnidir::projectPoints(obj, img, rvec, tvec, Ktrue, xiTrue, Dtrue);

        // Validate: finite + inside image
        bool ok = true;
        double maxR = 0.0;

        for (const auto& p : img)
        {
           if (!std::isfinite(p.x) || !std::isfinite(p.y))
          {
              ok = false; break;
          }

            if (p.x < margin || p.y < margin ||
                p.x > imageSize.width  - 1 - margin ||
                p.y > imageSize.height - 1 - margin)
            {
                ok = false; break;
            }

            const double dx = (double)p.x - cx;
            const double dy = (double)p.y - cy;
            maxR = std::max(maxR, std::sqrt(dx*dx + dy*dy));
        }

        // Enforce radius coverage: need points close to edges
        // (tune threshold if needed)
        const double desiredR = 0.38 * std::min(imageSize.width, imageSize.height);
        if (!ok || maxR < desiredR)
            continue;

        // Add pixel noise
        for (auto& p : img)
        {
            p.x += (float)rng.gaussian(noiseSigmaPx);
            p.y += (float)rng.gaussian(noiseSigmaPx);
        }

        objectPoints.push_back(obj);
        imagePoints.push_back(std::move(img));
    }

    if ((int)imagePoints.size() < nViews)
    {
        CV_Error(cv::Error::StsError,
                 "synthesizeOmniData: Could not generate enough valid views (k3 observable). "
                 "Try increasing maxTries or relaxing constraints.");
    }
}

/**
 * Real-data sanity for 5-parameter omnidir model (includes k3):
 * Runs calibration on opencv_extra stereo/case1 images and verifies parameters are finite/reasonable,
 * that undistort/rectify maps can be constructed, and that external reprojection RMS is consistent.
 */
TEST(Omnidir_K3_Hard, RealData_5params_K3_Free_StabilityAndSanity)
{
    using namespace cv;

    const Size board(9,6);
    const float square = 0.04f;

    std::vector<std::vector<Point3f>> objPts;
    std::vector<std::vector<Point2f>> imgPts;
    Size imageSize;
    loadRealData(board, square, objPts, imgPts, imageSize);

    // 5-parameter model (k3 included)
    Mat K  = Mat::eye(3,3,CV_64F);
    Mat xi(1,1,CV_64F); xi.at<double>(0,0) = 1.0;
    Mat D5(1,5,CV_64F, Scalar(0));

    std::vector<Mat> rvecs, tvecs;
    Mat idx;

    const int flags = omnidir::CALIB_FIX_SKEW; // keep it comparable to your other tests
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 1e-9);

    double rms = omnidir::calibrate(objPts, imgPts, imageSize, K, xi, D5,
                                    rvecs, tvecs, flags, criteria, idx);

    // Must have used some views
    ASSERT_GT((int)idx.total(), 0);
    ASSERT_EQ((int)D5.total(), 5);

    // maps should be constructible with 5-param distortion (k3 included)
    Mat xiMat(1,1,CV_64F); 
    xiMat.at<double>(0,0) = xi.at<double>(0,0);
    Mat map1, map2;
    omnidir::initUndistortRectifyMap(
        K, D5, xiMat, Matx33d::eye(), K, imageSize,
        CV_32FC1, map1, map2,
        omnidir::RECTIFY_PERSPECTIVE);

    EXPECT_FALSE(map1.empty());
    EXPECT_FALSE(map2.empty());


    // RMS sanity (broad but meaningful)
    EXPECT_TRUE(std::isfinite(rms));
    EXPECT_GT(rms, 0.0);
    EXPECT_LT(rms, 5.0);

    // Intrinsics sanity
    const double fx = K.at<double>(0,0);
    const double fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2);
    const double cy = K.at<double>(1,2);
    const double xiv = xi.at<double>(0,0);

    EXPECT_TRUE(std::isfinite(fx));
    EXPECT_TRUE(std::isfinite(fy));
    EXPECT_TRUE(std::isfinite(cx));
    EXPECT_TRUE(std::isfinite(cy));
    EXPECT_TRUE(std::isfinite(xiv));

    EXPECT_GT(fx, 0.0);
    EXPECT_GT(fy, 0.0);
    EXPECT_GE(cx, 0.0);
    EXPECT_GE(cy, 0.0);
    EXPECT_LT(cx, imageSize.width);
    EXPECT_LT(cy, imageSize.height);

    // xi should not explode
    EXPECT_LT(std::abs(xiv), 10.0);

    // Distortion sanity (all finite, not insane)
    Mat d = D5.reshape(1,1);
    for (int i = 0; i < 5; ++i)
    {
        const double v = d.at<double>(0,i);
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LT(std::abs(v), 100.0);
    }

    // External reprojection RMS cross-check (optional but good)
    const double reproj = computeReprojRms(objPts, imgPts, K, xiv, D5, rvecs, tvecs, idx);
    EXPECT_TRUE(std::isfinite(reproj));
    EXPECT_GT(reproj, 0.0);
    EXPECT_LT(reproj, 5.0);

    // Reproj RMS should be close-ish to internal RMS (loose tolerance)
    EXPECT_LE(std::abs(reproj - rms), std::max(0.05, 0.02 * rms));
}

/**
 * Real-data regression (4-parameter legacy model):
 * Calibrates with 4 distortion coefficients on opencv_extra stereo/case1 and compares K/xi/D4
 * against pre-recorded "golden" values to catch unintended behavior changes (backward stability).
 */
TEST(Omnidir_K3_Hard, RealData_GoldenCoefficients_4params)
{
    const Size board(9,6);
    const float square = 0.04f;

    vector<vector<Point3f>> objPts;
    vector<vector<Point2f>> imgPts;
    Size imageSize;

    loadRealData(board, square, objPts, imgPts, imageSize);

    Mat K  = Mat::eye(3,3,CV_64F);
    Mat xi = Mat(1,1,CV_64F); xi.at<double>(0,0) = 1.0;
    Mat D4 = Mat(1,4,CV_64F, Scalar(0));

    vector<Mat> r, t;
    Mat idx;

    const int flags = omnidir::CALIB_FIX_SKEW;
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 250, 1e-8);

    double rms = omnidir::calibrate(objPts, imgPts, imageSize, K, xi, D4, r, t, flags, criteria, idx);

    ASSERT_GT((int)idx.total(), 0);
    ASSERT_EQ((int)D4.total(), 4);

    // --- GOLDEN VALUES (FILL THESE FROM BASELINE RUN) ---
    // Run once, print K/xi/D4, then paste here.
    // Suggested printing (temporarily):
    // std::cout << "K=\n" << K << "\nxi=" << xi << "\nD4=" << D4 << "\nRMS=" << rms << std::endl;

    const double GOLD_fx = 978.1963596766802;
    const double GOLD_fy = 978.1142764489673;
    const double GOLD_cx = 342.3670276970017;
    const double GOLD_cy = 235.6861159367739;
    const double GOLD_xi = 0.8234519365247485;
    const double GOLD_d0 = -0.1760250991330968;
    const double GOLD_d1 = -0.8070188169070457;
    const double GOLD_d2 = 0.003591801582976898;
    const double GOLD_d3 = -0.000695840037846224;

    // Tolerances: keep stable across platforms
    const double TOL_f  = 8.0;   // pixels
    const double TOL_c  = 8.0;   // pixels
    const double TOL_xi = 0.08;  // unitless
    const double TOL_d  = 0.05;  // distortion coeffs

    EXPECT_NEAR(K.at<double>(0,0), GOLD_fx, TOL_f);
    EXPECT_NEAR(K.at<double>(1,1), GOLD_fy, TOL_f);
    EXPECT_NEAR(K.at<double>(0,2), GOLD_cx, TOL_c);
    EXPECT_NEAR(K.at<double>(1,2), GOLD_cy, TOL_c);
    EXPECT_NEAR(xi.at<double>(0,0), GOLD_xi, TOL_xi);
    EXPECT_NEAR(D4.at<double>(0,0), GOLD_d0, TOL_d);
    EXPECT_NEAR(D4.at<double>(0,1), GOLD_d1, TOL_d);
    EXPECT_NEAR(D4.at<double>(0,2), GOLD_d2, TOL_d);
    EXPECT_NEAR(D4.at<double>(0,3), GOLD_d3, TOL_d);

    // Sanity: RMS reasonable and finite.
    EXPECT_TRUE(std::isfinite(rms));
    EXPECT_LT(rms, 5.0);
}

/**
 * Backward compatibility check when k3==0:
 * Verifies that using D4 and D5=[D4,k3=0] produces identical:
 *  - projections (projectPoints),
 *  - undistorted points (undistortPoints),
 *  - undistort/rectify maps (initUndistortRectifyMap),
 * within tight tolerances.
 */
TEST(Omnidir_K3_Hard, BackwardCompat_4coeffs_equals_5coeffs_k3_zero_project_and_maps)
{
    using namespace cv;

    const Matx33d K(520, 0, 320,
                    0, 520, 240,
                    0,   0,   1);
    const double xi = 1.2;

    const Size sz(640, 480);
    Mat xiMat(1, 1, CV_64F);
    xiMat.at<double>(0, 0) = xi;

    Mat D4(1, 4, CV_64F);
    D4.at<double>(0, 0) = -0.10;
    D4.at<double>(0, 1) =  0.01;
    D4.at<double>(0, 2) =  0.001;
    D4.at<double>(0, 3) = -0.0005;

    Mat D5(1, 5, CV_64F);
    for (int i = 0; i < 4; ++i) D5.at<double>(0, i) = D4.at<double>(0, i);
    D5.at<double>(0, 4) = 0.0; // k3 = 0

    // --- projectPoints equality ---
    vector<Point3f> obj = makeChessboard3D(Size(9, 6), 0.04f);
    Mat rvec = (Mat_<double>(3, 1) << 0.2, -0.1, 0.05);
    Mat tvec = (Mat_<double>(3, 1) << 0.40, -0.25, 1.80);

    vector<Point2f> p4, p5;
    cv::omnidir::projectPoints(obj, p4, rvec, tvec, K, xi, D4);
    cv::omnidir::projectPoints(obj, p5, rvec, tvec, K, xi, D5);

    ASSERT_EQ(p4.size(), p5.size());

    double maxAbs = 0.0;
    for (size_t i = 0; i < p4.size(); ++i)
    {
        maxAbs = std::max(maxAbs, std::abs((double)p4[i].x - (double)p5[i].x));
        maxAbs = std::max(maxAbs, std::abs((double)p4[i].y - (double)p5[i].y));
    }
    EXPECT_LT(maxAbs, 1e-8);

    // --- undistortPoints equality (k3=0) ---
    // Use distorted points generated with the legacy (D4) model.
    vector<Point2f> distorted;
    cv::omnidir::projectPoints(obj, distorted, rvec, tvec, K, xi, D4);

    Mat und4, und5;
    cv::omnidir::undistortPoints(distorted, und4, K, D4, xiMat, Matx33d::eye());
    cv::omnidir::undistortPoints(distorted, und5, K, D5, xiMat, Matx33d::eye());

    ASSERT_FALSE(und4.empty());
    ASSERT_FALSE(und5.empty());
    ASSERT_EQ(und4.total(), und5.total());
    ASSERT_EQ(und4.type(), und5.type());

    EXPECT_LT(maxAbsDiff(und4, und5), 1e-8);

    // --- initUndistortRectifyMap equality ---
    Mat m1_4, m2_4, m1_5, m2_5;
    cv::omnidir::initUndistortRectifyMap(K, D4, xiMat, Matx33d::eye(), K, sz,
                                         CV_32FC1, m1_4, m2_4,
                                         cv::omnidir::RECTIFY_PERSPECTIVE);
    cv::omnidir::initUndistortRectifyMap(K, D5, xiMat, Matx33d::eye(), K, sz,
                                         CV_32FC1, m1_5, m2_5,
                                         cv::omnidir::RECTIFY_PERSPECTIVE);

    ASSERT_FALSE(m1_4.empty());
    ASSERT_FALSE(m2_4.empty());
    ASSERT_FALSE(m1_5.empty());
    ASSERT_FALSE(m2_5.empty());

    EXPECT_LT(maxAbsDiff(m1_4, m1_5), 1e-6);
    EXPECT_LT(maxAbsDiff(m2_4, m2_5), 1e-6);
}

/**
 * Deterministic synthetic validation of k3 parameter:
 * Generates image points using known K/xi/Dtrue with non-zero k3, then calibrates with k3 fixed vs free.
 * Ensures free-k3 improves reprojection RMS and recovers k3 close to the ground-truth value.
 */
TEST(Omnidir_K3_Hard, Synthetic_Recovers_K3_and_Improves_Fit_vs_4params)
{
    const Size imageSize(640, 480);
    const Size board(9, 6);
    const float square = 0.04f;
    const int nViews = 20;

    const Matx33d Ktrue(520, 0, 320,
                        0, 520, 240,
                        0,   0,   1);
    const double xiTrue = 1.15;

    Mat Dtrue(1, 5, CV_64F);
    Dtrue.at<double>(0,0) = -0.10;   // k1
    Dtrue.at<double>(0,1) =  0.01;   // k2
    Dtrue.at<double>(0,2) =  0.001;  // p1
    Dtrue.at<double>(0,3) = -0.0005; // p2
    Dtrue.at<double>(0,4) = -0.2;    // k3

    vector<Point3f> obj = makeChessboard3D(board, square);
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;

    synthesizeOmniData(nViews, obj, imageSize, Ktrue, xiTrue, Dtrue,
                       objectPoints, imagePoints,
                       /*noiseSigmaPx=*/0.02, /*seed=*/0xCAFE);

    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 1e-9);

    const int flags_common =
        cv::omnidir::CALIB_USE_GUESS |
        cv::omnidir::CALIB_FIX_XI |
        cv::omnidir::CALIB_FIX_K1 |
        cv::omnidir::CALIB_FIX_K2 |
        cv::omnidir::CALIB_FIX_P1 |
        cv::omnidir::CALIB_FIX_P2 |
        cv::omnidir::CALIB_FIX_SKEW |
        cv::omnidir::CALIB_FIX_CENTER |
        cv::omnidir::CALIB_FIX_GAMMA;

    // ----------------------------
    // Run A: k3 FIXED to 0
    // ----------------------------
    Mat K_A  = Mat(Ktrue).clone();
    Mat xi_A(1,1,CV_64F); xi_A.at<double>(0,0) = xiTrue;

    Mat D_A = Dtrue.clone();
    D_A.at<double>(0,4) = 0.0;

    vector<Mat> rA, tA;
    Mat idxA;

    const int flags_A = flags_common | cv::omnidir::CALIB_FIX_K3;

    double rms_A = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize,
                                          K_A, xi_A, D_A, rA, tA,
                                          flags_A, criteria, idxA);

    // ----------------------------
    // Run B: k3 FREE (start from 0 and must recover)
    // ----------------------------
    Mat K_B  = Mat(Ktrue).clone();
    Mat xi_B(1,1,CV_64F); xi_B.at<double>(0,0) = xiTrue;

    Mat D_B = Dtrue.clone();
    D_B.at<double>(0,4) = 0.0;

    vector<Mat> rB, tB;
    Mat idxB;

    const int flags_B = flags_common; // no FIX_K3

    double rms_B = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize,
                                          K_B, xi_B, D_B, rB, tB,
                                          flags_B, criteria, idxB);

    ASSERT_GT((int)idxA.total(), 0);
    ASSERT_GT((int)idxB.total(), 0);

    // 1) internal RMS: B should not be worse than A (small slack)
    EXPECT_LE(rms_B, rms_A + 1e-12);

    // 2) external reprojection RMS: B must be better than A (strong check)
    const double reproj_A = computeReprojRms(objectPoints, imagePoints, K_A, xiTrue, D_A, rA, tA, idxA);
    const double reproj_B = computeReprojRms(objectPoints, imagePoints, K_B, xiTrue, D_B, rB, tB, idxB);
    EXPECT_LT(reproj_B, reproj_A - 1e-4);

    // k3 sanity + recovery
    Mat dB = D_B.reshape(1, 1);
    ASSERT_EQ((int)dB.total(), 5);

    const double k3_est = dB.at<double>(0,4);
    EXPECT_TRUE(std::isfinite(k3_est));
    EXPECT_LT(std::abs(k3_est), 5.0);
    EXPECT_NEAR(k3_est, Dtrue.at<double>(0,4), 0.2);
    EXPECT_GT(std::abs(k3_est), 1e-4);
}
} // namespace opencv_test
