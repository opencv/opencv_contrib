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
#include <map>
#include <string>
#include <cctype>

namespace opencv_test {

using namespace cv;
using std::vector;
using std::string;

// Returns the maximum absolute element-wise difference between two Mats.
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

//Generates the 3D chessboard corner coordinates for a given board size and square spacing.
static vector<Point3f> makeChessboard3D(Size board, float square)
{
    vector<Point3f> obj;
    obj.reserve((size_t)board.area());
    for (int y = 0; y < board.height; ++y)
        for (int x = 0; x < board.width; ++x)
            obj.emplace_back((float)x * square, (float)y * square, 0.f);
    return obj;
}

static std::string getTestDataRoot()
{
    std::string root = cvtest::TS::ptr()->get_data_path();
    if (!root.empty() && root.back() != '/')
    {
        root.push_back('/');
    }
    const char suffix[] = "ccalib/";
    constexpr size_t sufLen = sizeof(suffix) - 1; // 7
    if (root.size() >= sufLen && root.compare(root.size() - sufLen, sufLen, suffix) == 0)
    {
        root.erase(root.size() - sufLen);
    }
    return root;
}

static std::string getDataPath()
{
    return getTestDataRoot() + "cv/stereo/case1/";
}

//Detects chessboard corners.
static bool detectCorners(const Mat& gray, Size board, vector<Point2f>& corners)
{
    bool ok = findChessboardCorners(gray, board, corners,CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    if (!ok)
    {
        return false;
    }
    cornerSubPix(gray, corners,Size(11,11), Size(-1,-1),TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 1e-4));
    return true;
}


// Compute reprojection RMS using omnidir::projectPoints.
// If idx is non-empty, only the used views are evaluated (calibrate may reject some).
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
        {
            idxRow.convertTo(idxRow, CV_32S);
        }
    }
    double sse = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < nUsed; ++i)
    {
        const int view = useIdx ? idxRow.at<int>((int)i) : (int)i;
        vector<Point2f> proj;
        cv::omnidir::projectPoints(objectPoints[(size_t)view], proj, rvecs[i], tvecs[i], K, xi, D);
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

static void loadRealData(Size board, float square, vector<vector<Point3f>>& objectPoints, vector<vector<Point2f>>& imagePoints, Size& imageSize)
{
    const std::string dir = getDataPath();
    vector<String> files;
    cv::glob(dir + "left*.png", files, false);
    std::sort(files.begin(), files.end());
    if (files.size() < 5)
    {
        throw cvtest::SkipTestException(
            "opencv_extra stereo/case1 data not found (need >= 5 left*.png). "
            "Expected: <data_path>/cv/stereo/case1/left*.png");
    }
    objectPoints.clear();
    imagePoints.clear();
    imageSize = Size();
    const auto obj = makeChessboard3D(board, square);
    for (const auto& f : files)
    {
        Mat img = imread(f, IMREAD_GRAYSCALE);
        if (img.empty())
        {
            continue;
        }
        if (imageSize.empty())
        {
            imageSize = img.size();
        }
        vector<Point2f> corners;
        if (!detectCorners(img, board, corners))
        {
            continue;
        }
        objectPoints.push_back(obj);
        imagePoints.push_back(corners);
    }
    if (objectPoints.size() < 5)
    {
        throw cvtest::SkipTestException("Not enough valid chessboard detections in: " + dir);
    }
}

// Synthetic data generator (deterministic, k3 observable)
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
            if (p.x < margin || p.y < margin || p.x > imageSize.width  - 1 - margin || p.y > imageSize.height - 1 - margin)
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
        {
            continue;
        }
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

// Extract a numeric "index" from a filename like ".../left01.png" or ".../right12.png".
// We just take the last contiguous digit run in the stem.
static int extractLastNumberFromPath(const cv::String& path)
{
    // basename without extension
    string s(path.c_str());
    // Remove directory
    size_t slash = s.find_last_of("/\\");
    string base = (slash == string::npos) ? s : s.substr(slash + 1);
    // Remove extension
    size_t dot = base.find_last_of('.');
    string stem = (dot == string::npos) ? base : base.substr(0, dot);

    // Find last digit run
    int end = (int)stem.size() - 1;
    while (end >= 0 && !std::isdigit((unsigned char)stem[(size_t)end])) end--;
    if (end < 0) return -1;

    int start = end;
    while (start >= 0 && std::isdigit((unsigned char)stem[(size_t)start])) start--;
    start++;

    try {
        return std::stoi(stem.substr((size_t)start, (size_t)(end - start + 1)));
    } catch (...) {
        return -1;
    }
}

// Builds a map: index -> filepath for all files matching pattern.
static std::map<int, cv::String> indexFiles(const vector<cv::String>& files)
{
    std::map<int, cv::String> m;
    for (const auto& f : files)
    {
        int idx = extractLastNumberFromPath(f);
        if (idx < 0) continue;
        // If duplicates exist, keep first (or overwrite; doesn't matter much)
        if (!m.count(idx)) m[idx] = f;
    }
    return m;
}

// Loads stereo chessboard observations from opencv_extra testdata (case1).
// Pairs left/right images by the numeric index in the filename, detects corners in both,
// and fills objectPoints + imagePointsL/R for stereoCalibrate.
static void loadStereoRealDataPairs(Size board, float square,
                                    vector<vector<Point3f>>& objectPoints,
                                    vector<vector<Point2f>>& imagePointsL,
                                    vector<vector<Point2f>>& imagePointsR,
                                    Size& imageSizeL,
                                    Size& imageSizeR)
{
    const std::string dir = getDataPath();
    vector<cv::String> leftFiles, rightFiles;
    cv::glob(dir + "left*.png", leftFiles, false);
    cv::glob(dir + "right*.png", rightFiles, false);

    if (leftFiles.size() < 5 || rightFiles.size() < 5)
        throw cvtest::SkipTestException(
            "opencv_extra stereo/case1 data not found (need >= 5 left*.png and right*.png). "
            "Expected: <data_path>/cv/stereo/case1/left*.png and right*.png");

    auto L = indexFiles(leftFiles);
    auto R = indexFiles(rightFiles);

    // Intersect indices
    vector<int> common;
    common.reserve(std::min(L.size(), R.size()));
    for (const auto& kv : L)
    {
        if (R.count(kv.first))
        {
            common.push_back(kv.first);
        }
    }
    if (common.size() < 5)
    {
        throw cvtest::SkipTestException("Not enough left/right pairs with matching indices in: " + dir);
    }
    std::sort(common.begin(), common.end());
    objectPoints.clear();
    imagePointsL.clear();
    imagePointsR.clear();
    imageSizeL = Size();
    imageSizeR = Size();
    const auto obj = makeChessboard3D(board, square);
    for (int id : common)
    {
        Mat imgL = imread(L[id], IMREAD_GRAYSCALE);
        Mat imgR = imread(R[id], IMREAD_GRAYSCALE);
        if (imgL.empty() || imgR.empty())
        {
            continue;
        }
        if (imageSizeL.empty())
        {
            imageSizeL = imgL.size();
        }
        if (imageSizeR.empty())
        {
            imageSizeR = imgR.size();
        }
        // If your dataset is guaranteed same size, assert:
        if (imgL.size() != imageSizeL || imgR.size() != imageSizeR)
        {
            continue;
        }
        vector<Point2f> cL, cR;
        if (!detectCorners(imgL, board, cL)) continue;
        if (!detectCorners(imgR, board, cR)) continue;

        objectPoints.push_back(obj);
        imagePointsL.push_back(std::move(cL));
        imagePointsR.push_back(std::move(cR));
    }
    if (objectPoints.size() < 5)
        throw cvtest::SkipTestException("Not enough valid stereo chessboard detections in: " + dir);
}

// Synthetic stereo generator (deterministic).
// Produces matched left/right observations for stereoCalibrate() with k3 observable
// (forces large radius coverage) + optional Gaussian noise.
static void synthesizeStereoOmniData(int nViews,
                                     const std::vector<cv::Point3f>& obj,
                                     const cv::Size& imageSize,
                                     const cv::Matx33d& K1true, double xi1True, const cv::Mat& D1true,
                                     const cv::Matx33d& K2true, double xi2True, const cv::Mat& D2true,
                                     const cv::Mat& omRig, const cv::Mat& TRig,   // 3x1 om, 3x1 T (rig: cam2 = rig * cam1)
                                     std::vector<std::vector<cv::Point3f>>& objectPoints,
                                     std::vector<std::vector<cv::Point2f>>& imagePointsL,
                                     std::vector<std::vector<cv::Point2f>>& imagePointsR,
                                     double noiseSigmaPx = 0.25,
                                     uint64 seed = 0xBEEF)
{
    CV_Assert(omRig.total() == 3 && TRig.total() == 3);
    CV_Assert(D1true.total() == 5 && D2true.total() == 5);

    cv::RNG rng((uint64)seed);

    objectPoints.clear();
    imagePointsL.clear();
    imagePointsR.clear();

    objectPoints.reserve((size_t)nViews);
    imagePointsL.reserve((size_t)nViews);
    imagePointsR.reserve((size_t)nViews);

    const double cx1 = K1true(0,2), cy1 = K1true(1,2);
    const double cx2 = K2true(0,2), cy2 = K2true(1,2);

    const int margin = 8;
    int tries = 0;
    const int maxTries = 40000;

    // Precompute rig rotation matrix
    cv::Mat Rrig;
    cv::Rodrigues(omRig, Rrig);
    Rrig.convertTo(Rrig, CV_64F);
    cv::Mat Trig64 = TRig.reshape(1,3);
    Trig64.convertTo(Trig64, CV_64F);

    while ((int)imagePointsL.size() < nViews && tries++ < maxTries)
    {
        // Random board pose in LEFT camera coordinates
        cv::Vec3d r(rng.uniform(-0.35, 0.35),
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(-0.35, 0.35));
        cv::Vec3d t(rng.uniform(-0.25, 0.25),
                    rng.uniform(-0.25, 0.25),
                    rng.uniform(0.55, 0.95));

        cv::Mat rvecL(r), tvecL(t);

        // Derive RIGHT pose: Xc2 = Rrig * Xc1 + Trig
        cv::Mat RL;
        cv::Rodrigues(rvecL, RL);
        cv::Mat RR = Rrig * RL;
        cv::Mat tR = Rrig * tvecL.reshape(1,3) + Trig64;

        cv::Mat rvecR;
        cv::Rodrigues(RR, rvecR);

        std::vector<cv::Point2f> imgL, imgR;
        cv::omnidir::projectPoints(obj, imgL, rvecL, tvecL, K1true, xi1True, D1true);
        cv::omnidir::projectPoints(obj, imgR, rvecR, tR,    K2true, xi2True, D2true);

        bool ok = true;
        double maxR1 = 0.0, maxR2 = 0.0;

        for (size_t i = 0; i < imgL.size(); ++i)
        {
            const auto& pL = imgL[i];
            const auto& pR = imgR[i];

            if (!std::isfinite(pL.x) || !std::isfinite(pL.y) ||
                !std::isfinite(pR.x) || !std::isfinite(pR.y))
            { ok = false; break; }

            if (pL.x < margin || pL.y < margin ||
                pL.x > imageSize.width - 1 - margin ||
                pL.y > imageSize.height - 1 - margin)
            { ok = false; break; }

            if (pR.x < margin || pR.y < margin ||
                pR.x > imageSize.width - 1 - margin ||
                pR.y > imageSize.height - 1 - margin)
            { ok = false; break; }

            const double dx1 = (double)pL.x - cx1, dy1 = (double)pL.y - cy1;
            const double dx2 = (double)pR.x - cx2, dy2 = (double)pR.y - cy2;
            maxR1 = std::max(maxR1, std::sqrt(dx1*dx1 + dy1*dy1));
            maxR2 = std::max(maxR2, std::sqrt(dx2*dx2 + dy2*dy2));
        }

        // Need strong radius coverage (k3 observability) in BOTH cameras
        const double desiredR = 0.38 * std::min(imageSize.width, imageSize.height);
        if (!ok || maxR1 < desiredR || maxR2 < desiredR)
            continue;

        // Add noise (deterministic)
        for (auto& p : imgL) { p.x += (float)rng.gaussian(noiseSigmaPx); p.y += (float)rng.gaussian(noiseSigmaPx); }
        for (auto& p : imgR) { p.x += (float)rng.gaussian(noiseSigmaPx); p.y += (float)rng.gaussian(noiseSigmaPx); }

        objectPoints.push_back(obj);
        imagePointsL.push_back(std::move(imgL));
        imagePointsR.push_back(std::move(imgR));
    }

    if ((int)imagePointsL.size() < nViews)
    {
        CV_Error(cv::Error::StsError,
                 "synthesizeStereoOmniData: Could not generate enough valid stereo views (k3 observable).");
    }
}

// Real-data sanity test for the 5-parameter omnidir distortion model (k1,k2,p1,p2,k3):
// - Runs omnidir::calibrate on opencv_extra cv/stereo/case1 (left*.png) detections.
// - Checks returned parameters are finite and within broad sanity ranges.
// - Verifies undistort/rectify maps can be built with D(1x5).
// Cross-checks the reported RMS by reprojecting points with omnidir::projectPoints.
TEST(Omnidir_K3_Hard, RealData_5params_K3_Free_StabilityAndSanity)
{
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
    double rms = omnidir::calibrate(objPts, imgPts, imageSize, K, xi, D5, rvecs, tvecs, flags, criteria, idx);
    // Must have used some views
    ASSERT_GT((int)idx.total(), 0);
    ASSERT_EQ((int)D5.total(), 5);
    // maps should be constructible with 5-param distortion (k3 included)
    Mat xiMat(1,1,CV_64F);
    xiMat.at<double>(0,0) = xi.at<double>(0,0);
    Mat map1, map2;
    omnidir::initUndistortRectifyMap(K, D5, xiMat, Matx33d::eye(), K, imageSize, CV_32FC1, map1, map2, omnidir::RECTIFY_PERSPECTIVE);
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
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(std::isfinite(D5.at<double>(0,i)));
    }
    // External reprojection RMS cross-check (optional but good)
    const double reproj = computeReprojRms(objPts, imgPts, K, xiv, D5, rvecs, tvecs, idx);
    EXPECT_TRUE(std::isfinite(reproj));
    EXPECT_GT(reproj, 0.0);
    EXPECT_LT(reproj, 5.0);

    // Reproj RMS should be close-ish to internal RMS (loose tolerance)
    EXPECT_LE(std::abs(reproj - rms), std::max(0.05, 0.02 * rms));
}

// Real-data regression (4-parameter legacy model):
// Calibrates with 4 distortion coefficients on opencv_extra stereo/case1 and compares K/xi/D4
// against pre-recorded "golden" values to catch unintended behavior changes (backward stability).
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

// Backward compatibility check when k3==0:
// Verifies that using D4 and D5=[D4,k3=0] produces identical:
//  - projections (projectPoints),
//  - undistorted points (undistortPoints),
//  - undistort/rectify maps (initUndistortRectifyMap),
// within tight tolerances.
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
    cv::omnidir::initUndistortRectifyMap(K, D4, xiMat, Matx33d::eye(), K, sz, CV_32FC1, m1_4, m2_4, cv::omnidir::RECTIFY_PERSPECTIVE);
    cv::omnidir::initUndistortRectifyMap(K, D5, xiMat, Matx33d::eye(), K, sz, CV_32FC1, m1_5, m2_5, cv::omnidir::RECTIFY_PERSPECTIVE);

    ASSERT_FALSE(m1_4.empty());
    ASSERT_FALSE(m2_4.empty());
    ASSERT_FALSE(m1_5.empty());
    ASSERT_FALSE(m2_5.empty());

    EXPECT_LT(maxAbsDiff(m1_4, m1_5), 1e-6);
    EXPECT_LT(maxAbsDiff(m2_4, m2_5), 1e-6);
}

// Deterministic synthetic validation of k3 parameter:
// Generates image points using known K/xi/Dtrue with non-zero k3, then calibrates with k3 fixed vs free.
// Ensures free-k3 improves reprojection RMS and recovers k3 close to the ground-truth value.
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

    // Run A: k3 FIXED to 0
    Mat K_A  = Mat(Ktrue).clone();
    Mat xi_A(1,1,CV_64F); xi_A.at<double>(0,0) = xiTrue;

    Mat D_A = Dtrue.clone();
    D_A.at<double>(0,4) = 0.0;

    vector<Mat> rA, tA;
    Mat idxA;

    const int flags_A = flags_common | cv::omnidir::CALIB_FIX_K3;
    double rms_A = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize, K_A, xi_A, D_A, rA, tA, flags_A, criteria, idxA);

    // Run B: k3 FREE (start from 0 and must recover)
    Mat K_B  = Mat(Ktrue).clone();
    Mat xi_B(1,1,CV_64F); xi_B.at<double>(0,0) = xiTrue;
    Mat D_B = Dtrue.clone();
    D_B.at<double>(0,4) = 0.0;
    vector<Mat> rB, tB;
    Mat idxB;
    const int flags_B = flags_common; // no FIX_K3
    double rms_B = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize, K_B, xi_B, D_B, rB, tB, flags_B, criteria, idxB);
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

// Stereo backward compatibility (real data):
// D5 with k3 fixed to 0 should behave similarly to the legacy D4 model.
// We check that calibration succeeds and yields comparable RMS and (k1,k2,p1,p2).
TEST(Omnidir_K3_Hard, Stereo_RealData_BackwardCompat_D4_equals_D5_k3_zero)
{
    const Size board(9,6);
    const float square = 0.04f;
    vector<vector<Point3f>> objPts;
    vector<vector<Point2f>> imgL, imgR;
    Size szL, szR;
    loadStereoRealDataPairs(board, square, objPts, imgL, imgR, szL, szR);

    // --- Run A: legacy D4 model ---
    Mat K1_A = Mat::eye(3,3,CV_64F);
    Mat K2_A = Mat::eye(3,3,CV_64F);
    Mat xi1_A(1,1,CV_64F); xi1_A.at<double>(0,0) = 1.0;
    Mat xi2_A(1,1,CV_64F); xi2_A.at<double>(0,0) = 1.0;

    Mat D1_A(1,4,CV_64F, Scalar(0));
    Mat D2_A(1,4,CV_64F, Scalar(0));

    Mat om_A, T_A;
    vector<Mat> rvecsL_A, tvecsL_A;
    Mat idx_A;

    const int flagsA = cv::omnidir::CALIB_FIX_SKEW;
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 250, 1e-8);

    double rms_A = cv::omnidir::stereoCalibrate(objPts, imgL, imgR, szL, szR,
                                                K1_A, xi1_A, D1_A,
                                                K2_A, xi2_A, D2_A,
                                                om_A, T_A, rvecsL_A, tvecsL_A,
                                                flagsA, criteria, idx_A);

    ASSERT_GT((int)idx_A.total(), 0);
    ASSERT_EQ((int)D1_A.total(), 4);
    ASSERT_EQ((int)D2_A.total(), 4);

    // --- Run B: D5 model but k3 FIXED to 0 ---
    Mat K1_B = Mat::eye(3,3,CV_64F);
    Mat K2_B = Mat::eye(3,3,CV_64F);
    Mat xi1_B(1,1,CV_64F); xi1_B.at<double>(0,0) = 1.0;
    Mat xi2_B(1,1,CV_64F); xi2_B.at<double>(0,0) = 1.0;

    Mat D1_B(1,5,CV_64F, Scalar(0));
    Mat D2_B(1,5,CV_64F, Scalar(0));
    D1_B.at<double>(0,4) = 0.0; // k3 fixed to 0
    D2_B.at<double>(0,4) = 0.0;

    Mat om_B, T_B;
    vector<Mat> rvecsL_B, tvecsL_B;
    Mat idx_B;

    const int flagsB = cv::omnidir::CALIB_FIX_SKEW | cv::omnidir::CALIB_FIX_K3;

    double rms_B = cv::omnidir::stereoCalibrate(objPts, imgL, imgR, szL, szR,
                                                K1_B, xi1_B, D1_B,
                                                K2_B, xi2_B, D2_B,
                                                om_B, T_B, rvecsL_B, tvecsL_B,
                                                flagsB, criteria, idx_B);

    ASSERT_GT((int)idx_B.total(), 0);
    ASSERT_EQ((int)D1_B.total(), 5);
    ASSERT_EQ((int)D2_B.total(), 5);

    // --- Expectations (loose but meaningful) ---
    EXPECT_TRUE(std::isfinite(rms_A));
    EXPECT_TRUE(std::isfinite(rms_B));
    EXPECT_LT(rms_A, 10.0);
    EXPECT_LT(rms_B, 10.0);

    // RMS should be close-ish (since model is effectively identical)
    EXPECT_LT(std::abs(rms_A - rms_B), 0.5);

    // k3 must remain exactly (or nearly) zero because FIX_K3 is set
    EXPECT_NEAR(D1_B.at<double>(0,4), 0.0, 1e-12);
    EXPECT_NEAR(D2_B.at<double>(0,4), 0.0, 1e-12);

    // Distortion first 4 coefficients should be comparable (loose tolerance)
    const double TOL_D = 0.3;
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_LT(std::abs(D1_A.at<double>(0,i) - D1_B.at<double>(0,i)), TOL_D);
        EXPECT_LT(std::abs(D2_A.at<double>(0,i) - D2_B.at<double>(0,i)), TOL_D);
    }
}

// Stereo real-data sanity with 5-coefficient distortion (includes k3, free):
// Verifies calibration produces finite parameters, reasonable RMS, and that rectification maps can be created.
TEST(Omnidir_K3_Hard, Stereo_RealData_5params_K3_Free_SanityAndRectifyMaps)
{
    const Size board(9,6);
    const float square = 0.04f;

    vector<vector<Point3f>> objPts;
    vector<vector<Point2f>> imgL, imgR;
    Size szL, szR;
    loadStereoRealDataPairs(board, square, objPts, imgL, imgR, szL, szR);

    Mat K1 = Mat::eye(3,3,CV_64F);
    Mat K2 = Mat::eye(3,3,CV_64F);
    Mat xi1(1,1,CV_64F); xi1.at<double>(0,0) = 1.0;
    Mat xi2(1,1,CV_64F); xi2.at<double>(0,0) = 1.0;

    Mat D1(1,5,CV_64F, Scalar(0));
    Mat D2(1,5,CV_64F, Scalar(0));

    Mat om, T;
    vector<Mat> rvecsL, tvecsL;
    Mat idx;

    const int flags = cv::omnidir::CALIB_FIX_SKEW;
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 1e-9);

    double rms = cv::omnidir::stereoCalibrate(objPts, imgL, imgR, szL, szR,
                                              K1, xi1, D1,
                                              K2, xi2, D2,
                                              om, T, rvecsL, tvecsL,
                                              flags, criteria, idx);

    ASSERT_GT((int)idx.total(), 0);
    ASSERT_EQ((int)D1.total(), 5);
    ASSERT_EQ((int)D2.total(), 5);

    EXPECT_TRUE(std::isfinite(rms));
    EXPECT_GT(rms, 0.0);
    EXPECT_LT(rms, 10.0);

    // Check finiteness (do NOT assert small magnitude for k3 on real data)
    auto finiteMat = [](const Mat& m){
        Mat mm = m.reshape(1,1);
        for (int i = 0; i < (int)mm.total(); ++i)
            if (!std::isfinite(mm.at<double>(0,i))) return false;
        return true;
    };

    EXPECT_TRUE(finiteMat(K1));
    EXPECT_TRUE(finiteMat(K2));
    EXPECT_TRUE(std::isfinite(xi1.at<double>(0,0)));
    EXPECT_TRUE(std::isfinite(xi2.at<double>(0,0)));

    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(std::isfinite(D1.at<double>(0,i)));
        EXPECT_TRUE(std::isfinite(D2.at<double>(0,i)));
    }

    // Rectify rotations
    Mat R1, R2;
    cv::omnidir::stereoRectify(om, T, R1, R2);
    EXPECT_FALSE(R1.empty());
    EXPECT_FALSE(R2.empty());

    // Build undistort/rectify maps for both cameras using D5
    Mat map1L, map2L, map1R, map2R;
    cv::omnidir::initUndistortRectifyMap(K1, D1, xi1, R1, K1, szL,
                                         CV_32FC1, map1L, map2L,
                                         cv::omnidir::RECTIFY_PERSPECTIVE);
    cv::omnidir::initUndistortRectifyMap(K2, D2, xi2, R2, K2, szR,
                                         CV_32FC1, map1R, map2R,
                                         cv::omnidir::RECTIFY_PERSPECTIVE);

    EXPECT_FALSE(map1L.empty());
    EXPECT_FALSE(map2L.empty());
    EXPECT_FALSE(map1R.empty());
    EXPECT_FALSE(map2R.empty());
}

// Synthetic stereo k3 check:
// Generate data with non-zero k3, then stereoCalibrate with k3 fixed vs free.
// Expect free-k3 to fit no worse and recover k3 towards ground truth.
TEST(Omnidir_K3_Hard, Stereo_Synthetic_Recovers_K3_and_Improves_Fit)
{
    const cv::Size imageSize(640, 480);
    const cv::Size board(9, 6);
    const float square = 0.04f;
    const int nViews = 20;

    // Ground-truth intrinsics
    const cv::Matx33d K1true(520, 0, 320,
                             0, 520, 240,
                             0,   0,   1);
    const cv::Matx33d K2true(525, 0, 320,
                             0, 525, 240,
                             0,   0,   1);

    const double xi1True = 1.10;
    const double xi2True = 1.15;

    // Ground-truth distortion (k3 non-zero)
    cv::Mat D1true(1,5,CV_64F), D2true(1,5,CV_64F);
    D1true.at<double>(0,0) = -0.10;  D1true.at<double>(0,1) = 0.01;
    D1true.at<double>(0,2) =  0.001; D1true.at<double>(0,3) = -0.0005;
    D1true.at<double>(0,4) = -0.20;  // k3

    D2true.at<double>(0,0) = -0.08;  D2true.at<double>(0,1) = 0.015;
    D2true.at<double>(0,2) =  0.0005;D2true.at<double>(0,3) = -0.0002;
    D2true.at<double>(0,4) = -0.18;  // k3

    // Stereo rig transform (cam2 w.r.t cam1)
    cv::Mat omRig = (cv::Mat_<double>(3,1) << 0.01, -0.02, 0.015);
    cv::Mat TRig  = (cv::Mat_<double>(3,1) << 0.10,  0.00, 0.00); // baseline ~10cm

    const auto obj = makeChessboard3D(board, square);

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imgL, imgR;

    synthesizeStereoOmniData(nViews, obj, imageSize,
                             K1true, xi1True, D1true,
                             K2true, xi2True, D2true,
                             omRig, TRig,
                             objectPoints, imgL, imgR,
                             /*noiseSigmaPx=*/0.02, /*seed=*/0xCAFE);

    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 1e-9);
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

    // --- Run A: k3 FIXED to 0 ---
    cv::Mat K1_A = cv::Mat(K1true).clone();
    cv::Mat K2_A = cv::Mat(K2true).clone();
    cv::Mat xi1_A(1,1,CV_64F); xi1_A.at<double>(0,0) = xi1True;
    cv::Mat xi2_A(1,1,CV_64F); xi2_A.at<double>(0,0) = xi2True;

    cv::Mat D1_A = D1true.clone(); D1_A.at<double>(0,4) = 0.0;
    cv::Mat D2_A = D2true.clone(); D2_A.at<double>(0,4) = 0.0;

    cv::Mat om_A, T_A;
    std::vector<cv::Mat> rA, tA;
    cv::Mat idxA;

    const int flags_A = flags_common | cv::omnidir::CALIB_FIX_K3;
    double rms_A = cv::omnidir::stereoCalibrate(objectPoints, imgL, imgR, imageSize, imageSize,
                                                K1_A, xi1_A, D1_A,
                                                K2_A, xi2_A, D2_A,
                                                om_A, T_A, rA, tA,
                                                flags_A, criteria, idxA);

    // --- Run B: k3 FREE (start from 0) ---
    cv::Mat K1_B = cv::Mat(K1true).clone();
    cv::Mat K2_B = cv::Mat(K2true).clone();
    cv::Mat xi1_B(1,1,CV_64F); xi1_B.at<double>(0,0) = xi1True;
    cv::Mat xi2_B(1,1,CV_64F); xi2_B.at<double>(0,0) = xi2True;

    cv::Mat D1_B = D1true.clone(); D1_B.at<double>(0,4) = 0.0;
    cv::Mat D2_B = D2true.clone(); D2_B.at<double>(0,4) = 0.0;

    cv::Mat om_B, T_B;
    std::vector<cv::Mat> rB, tB;
    cv::Mat idxB;

    const int flags_B = flags_common; // k3 free
    double rms_B = cv::omnidir::stereoCalibrate(objectPoints, imgL, imgR, imageSize, imageSize,
                                                K1_B, xi1_B, D1_B,
                                                K2_B, xi2_B, D2_B,
                                                om_B, T_B, rB, tB,
                                                flags_B, criteria, idxB);

    ASSERT_GT((int)idxA.total(), 0);
    ASSERT_GT((int)idxB.total(), 0);

    // Free-k3 should not be worse (small slack)
    EXPECT_LE(rms_B, rms_A + 1e-12);

    // k3 should move away from 0 and towards ground truth (loose tolerances)
    EXPECT_GT(std::abs(D1_B.at<double>(0,4)), 1e-4);
    EXPECT_GT(std::abs(D2_B.at<double>(0,4)), 1e-4);
    EXPECT_NEAR(D1_B.at<double>(0,4), D1true.at<double>(0,4), 0.25);
    EXPECT_NEAR(D2_B.at<double>(0,4), D2true.at<double>(0,4), 0.25);
}

// Stereo backward compatibility at the mapping level:
// When k3 == 0, using D4 and using D5=[D4,0] should yield identical
// undistort/rectify maps (up to small floating differences).
// Why: This directly validates the k3 integration inside map generation for stereo flows.
TEST(Omnidir_K3_Hard, Stereo_BackwardCompat_D4_equals_D5_k3_zero_rectify_maps)
{
    const cv::Matx33d K(520, 0, 320,
                        0, 520, 240,
                        0,   0,   1);
    const double xi = 1.2;
    const cv::Size sz(640, 480);

    cv::Mat xiMat(1,1,CV_64F); xiMat.at<double>(0,0) = xi;

    cv::Mat D4(1,4,CV_64F);
    D4.at<double>(0,0) = -0.10;
    D4.at<double>(0,1) =  0.01;
    D4.at<double>(0,2) =  0.001;
    D4.at<double>(0,3) = -0.0005;

    cv::Mat D5(1,5,CV_64F, cv::Scalar(0));
    for (int i = 0; i < 4; ++i) D5.at<double>(0,i) = D4.at<double>(0,i);
    D5.at<double>(0,4) = 0.0;

    // Use some plausible stereo rectify rotations (not necessarily from real calibration)
    cv::Mat om = (cv::Mat_<double>(3,1) << 0.01, -0.02, 0.015);
    cv::Mat T  = (cv::Mat_<double>(3,1) << 0.10,  0.00, 0.00);

    cv::Mat R1, R2;
    cv::omnidir::stereoRectify(om, T, R1, R2);

    cv::Mat m1L_4, m2L_4, m1L_5, m2L_5;
    cv::Mat m1R_4, m2R_4, m1R_5, m2R_5;

    cv::omnidir::initUndistortRectifyMap(K, D4, xiMat, R1, K, sz, CV_32FC1, m1L_4, m2L_4, cv::omnidir::RECTIFY_PERSPECTIVE);
    cv::omnidir::initUndistortRectifyMap(K, D5, xiMat, R1, K, sz, CV_32FC1, m1L_5, m2L_5, cv::omnidir::RECTIFY_PERSPECTIVE);

    cv::omnidir::initUndistortRectifyMap(K, D4, xiMat, R2, K, sz, CV_32FC1, m1R_4, m2R_4, cv::omnidir::RECTIFY_PERSPECTIVE);
    cv::omnidir::initUndistortRectifyMap(K, D5, xiMat, R2, K, sz, CV_32FC1, m1R_5, m2R_5, cv::omnidir::RECTIFY_PERSPECTIVE);

    ASSERT_FALSE(m1L_4.empty()); ASSERT_FALSE(m2L_4.empty());
    ASSERT_FALSE(m1L_5.empty()); ASSERT_FALSE(m2L_5.empty());
    ASSERT_FALSE(m1R_4.empty()); ASSERT_FALSE(m2R_4.empty());
    ASSERT_FALSE(m1R_5.empty()); ASSERT_FALSE(m2R_5.empty());

    EXPECT_LT(maxAbsDiff(m1L_4, m1L_5), 1e-6);
    EXPECT_LT(maxAbsDiff(m2L_4, m2L_5), 1e-6);
    EXPECT_LT(maxAbsDiff(m1R_4, m1R_5), 1e-6);
    EXPECT_LT(maxAbsDiff(m2R_4, m2R_5), 1e-6);
}

// Stereo real-data regression (legacy D4: k1,k2,p1,p2):
// Runs omnidir::stereoCalibrate on opencv_extra cv/stereo/case1 and compares K/xi/D/om/T/RMS
// against recorded golden values to detect unintended changes (backward compatibility).
TEST(Omnidir_K3_Hard, Stereo_RealData_GoldenCoefficients_4params)
{
    const cv::Size board(9,6);
    const float square = 0.04f;

    std::vector<std::vector<cv::Point3f>> objPts;
    std::vector<std::vector<cv::Point2f>> imgL, imgR;
    cv::Size szL, szR;
    loadStereoRealDataPairs(board, square, objPts, imgL, imgR, szL, szR);

    cv::Mat K1 = cv::Mat::eye(3,3,CV_64F);
    cv::Mat K2 = cv::Mat::eye(3,3,CV_64F);
    cv::Mat xi1(1,1,CV_64F); xi1.at<double>(0,0) = 1.0;
    cv::Mat xi2(1,1,CV_64F); xi2.at<double>(0,0) = 1.0;

    cv::Mat D1(1,4,CV_64F, cv::Scalar(0));
    cv::Mat D2(1,4,CV_64F, cv::Scalar(0));

    cv::Mat om, T;
    std::vector<cv::Mat> rvecsL, tvecsL;
    cv::Mat idx;

    const int flags = cv::omnidir::CALIB_FIX_SKEW;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 1e-9);

    const double rms = cv::omnidir::stereoCalibrate(
        objPts, imgL, imgR, szL, szR,
        K1, xi1, D1,
        K2, xi2, D2,
        om, T, rvecsL, tvecsL,
        flags, criteria, idx);

    ASSERT_GT((int)idx.total(), 0);
    ASSERT_EQ((int)D1.total(), 4);
    ASSERT_EQ((int)D2.total(), 4);
    ASSERT_TRUE(std::isfinite(rms));

    // ---- GOLDEN VALUES (fill from baseline run) ----
    // Run once with prints and paste here:
    // std::cout << "K1=\n" << K1 << "\nxi1=" << xi1 << "\nD1=" << D1 << "\n";
    // std::cout << "K2=\n" << K2 << "\nxi2=" << xi2 << "\nD2=" << D2 << "\n";
    // std::cout << "om=\n" << om << "\nT=\n" << T << "\nRMS=" << rms << "\n";

    const double GOLD_rms = 0.444818;
    const double GOLD_K1_fx = 1401.192283542995;
    const double GOLD_K1_fy = 1400.812329013437;
    const double GOLD_K1_cx = 342.3254062702503;
    const double GOLD_K1_cy = 235.2417898096324;
    const double GOLD_xi1 = 1.614168743335463;
    const double GOLD_D1_0 = 0.2235385579731314;
    const double GOLD_D1_1 = -4.476668984126793;
    const double GOLD_D1_2 = 0.005133958683733851;
    const double GOLD_D1_3 = -0.0009732332635782253;
    const double GOLD_K2_fx = 550.0137369584638;
    const double GOLD_K2_fy = 549.4958410910596;
    const double GOLD_K2_cx = 328.2231527771337;
    const double GOLD_K2_cy = 248.8295934261545;
    const double GOLD_xi2 = 0.01924461672982071;
    const double GOLD_D2_0 = -0.2803838371535953;
    const double GOLD_D2_1 = 0.09113955760046222;
    const double GOLD_D2_2 = -0.0004255779307760857;
    const double GOLD_D2_3 = 0.001086841676649092;
    const double GOLD_om_0 = 0.004178062404200768;
    const double GOLD_om_1 = 0.003073878685088316;
    const double GOLD_om_2 = -0.003814585954328772;
    const double GOLD_T_0  = -0.13351577905041;
    const double GOLD_T_1  = 0.001543343941214281;
    const double GOLD_T_2  = -4.292093079023809e-05;
    // ---- Tolerances (start here; relax if needed for CI stability) ----
    const double TOL_f   = 25.0;   // pixels
    const double TOL_c   = 25.0;   // pixels
    const double TOL_xi  = 0.25;   // unitless
    const double TOL_d   = 0.40;   // distortion
    const double TOL_om  = 0.35;   // radians-ish (rotation vector magnitude scale)
    const double TOL_T   = 0.25;   // translation in "board units" scale (depends on square)
    const double TOL_rms = 0.8;

    EXPECT_NEAR(rms, GOLD_rms, TOL_rms);

    EXPECT_NEAR(K1.at<double>(0,0), GOLD_K1_fx, TOL_f);
    EXPECT_NEAR(K1.at<double>(1,1), GOLD_K1_fy, TOL_f);
    EXPECT_NEAR(K1.at<double>(0,2), GOLD_K1_cx, TOL_c);
    EXPECT_NEAR(K1.at<double>(1,2), GOLD_K1_cy, TOL_c);

    EXPECT_NEAR(K2.at<double>(0,0), GOLD_K2_fx, TOL_f);
    EXPECT_NEAR(K2.at<double>(1,1), GOLD_K2_fy, TOL_f);
    EXPECT_NEAR(K2.at<double>(0,2), GOLD_K2_cx, TOL_c);
    EXPECT_NEAR(K2.at<double>(1,2), GOLD_K2_cy, TOL_c);

    EXPECT_NEAR(xi1.at<double>(0,0), GOLD_xi1, TOL_xi);
    EXPECT_NEAR(xi2.at<double>(0,0), GOLD_xi2, TOL_xi);

    EXPECT_NEAR(D1.at<double>(0,0), GOLD_D1_0, TOL_d);
    EXPECT_NEAR(D1.at<double>(0,1), GOLD_D1_1, TOL_d);
    EXPECT_NEAR(D1.at<double>(0,2), GOLD_D1_2, TOL_d);
    EXPECT_NEAR(D1.at<double>(0,3), GOLD_D1_3, TOL_d);

    EXPECT_NEAR(D2.at<double>(0,0), GOLD_D2_0, TOL_d);
    EXPECT_NEAR(D2.at<double>(0,1), GOLD_D2_1, TOL_d);
    EXPECT_NEAR(D2.at<double>(0,2), GOLD_D2_2, TOL_d);
    EXPECT_NEAR(D2.at<double>(0,3), GOLD_D2_3, TOL_d);

    EXPECT_NEAR(om.at<double>(0,0), GOLD_om_0, TOL_om);
    EXPECT_NEAR(om.at<double>(1,0), GOLD_om_1, TOL_om);
    EXPECT_NEAR(om.at<double>(2,0), GOLD_om_2, TOL_om);

    EXPECT_NEAR(T.at<double>(0,0), GOLD_T_0, TOL_T);
    EXPECT_NEAR(T.at<double>(1,0), GOLD_T_1, TOL_T);
    EXPECT_NEAR(T.at<double>(2,0), GOLD_T_2, TOL_T);
}
} // namespace opencv_test