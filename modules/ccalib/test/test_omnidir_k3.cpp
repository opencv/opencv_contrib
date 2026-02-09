
#include "test_precomp.hpp"
namespace opencv_test {

using cv::Mat;
using cv::Matx33d;
using cv::Point2f;
using cv::Point3f;
using cv::RNG;
using cv::Size;
using std::vector;

static double maxAbsDiff(const cv::Mat &a, const cv::Mat &b) {
  CV_Assert(a.size() == b.size());
  CV_Assert(a.type() == b.type());
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  double maxv = 0.0;
  cv::minMaxLoc(diff.reshape(1), nullptr, &maxv);
  return maxv;
}

static vector<Point3f> makeChessboard3D(Size board, float square) {
  vector<Point3f> obj;
  obj.reserve((size_t)board.area());
  for (int y = 0; y < board.height; ++y)
    for (int x = 0; x < board.width; ++x)
      obj.emplace_back((float)x * square, (float)y * square, 0.0f);
  return obj;
}

// Synthetic data with true k3 != 0
static void synthesizeOmniData(int nViews, const vector<Point3f> &obj,
                               const Matx33d &Ktrue, double xiTrue,
                               const Mat &Dtrue,
                               vector<vector<Point3f>> &objectPoints,
                               vector<vector<Point2f>> &imagePoints,
                               double noiseSigmaPx = 0.5,
                               uint64 seed = 0xBEEF) {
  RNG rng((uint64)seed);

  objectPoints.assign((size_t)nViews, obj);
  imagePoints.resize((size_t)nViews);

  for (int i = 0; i < nViews; ++i) {
    cv::Vec3d r(rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                rng.uniform(-0.4, 0.4));
    cv::Vec3d t(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                rng.uniform(1.8, 3.0));

    Mat rvec(r), tvec(t);

    vector<Point2f> img;
    cv::omnidir::projectPoints(obj, img, rvec, tvec, Ktrue, xiTrue, Dtrue);

    for (auto &p : img) {
      p.x += (float)rng.gaussian(noiseSigmaPx);
      p.y += (float)rng.gaussian(noiseSigmaPx);
    }
    imagePoints[(size_t)i] = std::move(img);
  }
}

// IMPORTANT:
// calibrate() returns rvecs/tvecs only for the views that passed
// initialization. Those views are listed in idx (OutputArray idx in
// omnidir::calibrate). So we must compute RMS only over the used views and map
// each rvec/tvec to its view index.
static double
computeReprojRms(const vector<vector<Point3f>> &objectPoints,
                 const vector<vector<Point2f>> &imagePoints, const Mat &K,
                 double xi, const Mat &D, const vector<Mat> &rvecs,
                 const vector<Mat> &tvecs,
                 const Mat &idx) // Nx1 or 1xN int indices; may be empty
{
  CV_Assert(objectPoints.size() == imagePoints.size());
  CV_Assert(rvecs.size() == tvecs.size());

  const bool useIdx = !idx.empty();
  const size_t nUsed = useIdx ? (size_t)idx.total() : objectPoints.size();

  CV_Assert(rvecs.size() == nUsed);

  double sse = 0.0;
  size_t n = 0;

  Mat idxRow;
  if (useIdx) {
    // Ensure int32, 1xN
    idxRow = idx.reshape(1, 1);
    if (idxRow.type() != CV_32S)
      idxRow.convertTo(idxRow, CV_32S);
  }

  for (size_t i = 0; i < nUsed; ++i) {
    const int view = useIdx ? idxRow.at<int>((int)i) : (int)i;

    vector<Point2f> proj;
    cv::omnidir::projectPoints(objectPoints[(size_t)view], proj, rvecs[i],
                               tvecs[i], K, xi, D);

    CV_Assert(proj.size() == imagePoints[(size_t)view].size());

    for (size_t j = 0; j < proj.size(); ++j) {
      const double dx =
          (double)proj[j].x - (double)imagePoints[(size_t)view][j].x;
      const double dy =
          (double)proj[j].y - (double)imagePoints[(size_t)view][j].y;
      sse += dx * dx + dy * dy;
    }
    n += proj.size();
  }

  return (n > 0) ? std::sqrt(sse / (double)n) : 0.0;
}

// Verifies that initUndistortRectifyMap accepts both 4- and 5-parameter
// distortion vectors (row/column shapes) without errors.

TEST(Omnidir_K3, InitUndistortRectifyMap_accepts_4_or_5_coeffs_and_shapes) {
  using namespace cv;

  const Size sz(640, 480);
  const Matx33d K(520, 0, 320, 0, 520, 240, 0, 0, 1);

  // Create xi as non-fixed Mat (avoid Mat_ wrapper surprises)
  Mat xiMat(1, 1, CV_64F);
  xiMat.at<double>(0, 0) = 1.2;

  Mat D4_row(1, 4, CV_64F);
  D4_row.at<double>(0, 0) = -0.1;
  D4_row.at<double>(0, 1) = 0.01;
  D4_row.at<double>(0, 2) = 0.0;
  D4_row.at<double>(0, 3) = 0.0;

  Mat D4_col = D4_row.t();

  Mat D5_row(1, 5, CV_64F);
  D5_row.at<double>(0, 0) = -0.1;
  D5_row.at<double>(0, 1) = 0.01;
  D5_row.at<double>(0, 2) = 0.0;
  D5_row.at<double>(0, 3) = 0.0;
  D5_row.at<double>(0, 4) = -0.006;

  Mat D5_col = D5_row.t();

  Mat map1, map2;

  // flags MUST be one of RECTIFY_* (0 is invalid and triggers assert)
  const int flags = omnidir::RECTIFY_PERSPECTIVE;

  omnidir::initUndistortRectifyMap(K, D4_row, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, map1, map2, flags);
  EXPECT_FALSE(map1.empty());
  EXPECT_FALSE(map2.empty());

  omnidir::initUndistortRectifyMap(K, D4_col, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, map1, map2, flags);
  EXPECT_FALSE(map1.empty());
  EXPECT_FALSE(map2.empty());

  omnidir::initUndistortRectifyMap(K, D5_row, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, map1, map2, flags);
  EXPECT_FALSE(map1.empty());
  EXPECT_FALSE(map2.empty());

  omnidir::initUndistortRectifyMap(K, D5_col, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, map1, map2, flags);
  EXPECT_FALSE(map1.empty());
  EXPECT_FALSE(map2.empty());
}

// Ensures backward compatibility: projecting points with 4 distortion
// coefficients matches 5 coefficients when k3 is set to zero.

TEST(Omnidir_K3,
     Backward_compat_4coeffs_same_as_5coeffs_with_k3_zero_for_projectPoints) {
  using namespace cv;

  const Matx33d K(520, 0, 320, 0, 520, 240, 0, 0, 1);
  const double xi = 1.2;

  Mat D4(1, 4, CV_64F);
  D4.at<double>(0, 0) = -0.1;
  D4.at<double>(0, 1) = 0.01;
  D4.at<double>(0, 2) = 0.001;
  D4.at<double>(0, 3) = -0.0005;

  Mat D5(1, 5, CV_64F);
  D5.at<double>(0, 0) = -0.1;
  D5.at<double>(0, 1) = 0.01;
  D5.at<double>(0, 2) = 0.001;
  D5.at<double>(0, 3) = -0.0005;
  D5.at<double>(0, 4) = 0.0; // k3 = 0

  vector<Point3f> obj = makeChessboard3D(Size(9, 6), 0.04f);

  Mat rvec = (Mat_<double>(3, 1) << 0.2, -0.1, 0.05);
  // offset + closer => larger radius, stronger check
  Mat tvec = (Mat_<double>(3, 1) << 0.40, -0.25, 1.80);

  vector<Point2f> p4, p5;
  omnidir::projectPoints(obj, p4, rvec, tvec, K, xi, D4);
  omnidir::projectPoints(obj, p5, rvec, tvec, K, xi, D5);

  ASSERT_EQ(p4.size(), p5.size());

  double maxAbs = 0.0;
  for (size_t i = 0; i < p4.size(); ++i) {
    maxAbs = std::max(maxAbs, std::abs((double)p4[i].x - (double)p5[i].x));
    maxAbs = std::max(maxAbs, std::abs((double)p4[i].y - (double)p5[i].y));
  }
  EXPECT_LT(maxAbs, 1e-8);
}

// Checks that changing only the k3 coefficient produces a measurable
// change in projected image points.

TEST(Omnidir_K3, ProjectPoints_changes_when_only_k3_changes) {
  using namespace cv;

  const Matx33d K(520, 0, 320, 0, 520, 240, 0, 0, 1);
  const double xi = 1.2;

  Mat D_k3_0(1, 5, CV_64F);
  D_k3_0.at<double>(0, 0) = -0.1;
  D_k3_0.at<double>(0, 1) = 0.01;
  D_k3_0.at<double>(0, 2) = 0.001;
  D_k3_0.at<double>(0, 3) = -0.0005;
  D_k3_0.at<double>(0, 4) = 0.0;

  Mat D_k3_1 = D_k3_0.clone();
  D_k3_1.at<double>(0, 4) =
      -0.2; // only k3 changes (stronger to avoid numerical ~0)

  vector<Point3f> obj = makeChessboard3D(Size(9, 6), 0.04f);

  Mat rvec = (Mat_<double>(3, 1) << 0.2, -0.1, 0.05);
  // Make points far from principal point in normalized space
  Mat tvec = (Mat_<double>(3, 1) << 0.45, -0.30, 1.80);

  vector<Point2f> p0, p1;
  omnidir::projectPoints(obj, p0, rvec, tvec, K, xi, D_k3_0);
  omnidir::projectPoints(obj, p1, rvec, tvec, K, xi, D_k3_1);

  ASSERT_EQ(p0.size(), p1.size());

  // RMS of pixel diffs (stable)
  double sse = 0.0;
  for (size_t i = 0; i < p0.size(); ++i) {
    const double dx = (double)p0[i].x - (double)p1[i].x;
    const double dy = (double)p0[i].y - (double)p1[i].y;
    sse += dx * dx + dy * dy;
  }
  const double rms = std::sqrt(sse / (double)p0.size());

  // If k3 is actually used, rms must be > 0.
  EXPECT_GT(rms, 1e-4);
  // sanity (not exploding)
  EXPECT_LT(rms, 200.0);
}

// Validates that calibration with 5 distortion parameters fits the data
// better (lower reprojection RMS) when the true model includes k3.

TEST(Omnidir_K3,Calibrate_5params_gives_lower_rms_than_4params_when_true_k3_nonzero) {
  using namespace cv;

  const Size imageSize(640, 480);
  const Size board(9, 6);
  const float square = 0.04f;
  const int nViews = 7;

  const Matx33d Ktrue(520, 0, 320, 0, 520, 240, 0, 0, 1);
  const double xiTrue = 1.15;

  Mat Dtrue(1, 5, CV_64F);
  Dtrue.at<double>(0, 0) = -0.10;
  Dtrue.at<double>(0, 1) = 0.01;
  Dtrue.at<double>(0, 2) = 0.001;
  Dtrue.at<double>(0, 3) = -0.0005;
  Dtrue.at<double>(0, 4) = -0.008; // true k3 != 0

  vector<Point3f> obj = makeChessboard3D(board, square);
  vector<vector<Point3f>> objectPoints;
  vector<vector<Point2f>> imagePoints;

  synthesizeOmniData(nViews, obj, Ktrue, xiTrue, Dtrue, objectPoints,
                     imagePoints, 0.5, 0xCAFE);

  // ---- Calibrate with 4 coeffs ----
  Mat K4 = Mat::eye(3, 3, CV_64F);
  Mat xi4(1, 1, CV_64F);
  xi4.at<double>(0, 0) = 1.0; // non-fixed Mat
  Mat D4(1, 4, CV_64F, Scalar(0));
  vector<Mat> r4, t4;

  // ---- Calibrate with 5 coeffs ----
  Mat K5 = Mat::eye(3, 3, CV_64F);
  Mat xi5(1, 1, CV_64F);
  xi5.at<double>(0, 0) = 1.0; // non-fixed Mat
  Mat D5(1, 5, CV_64F, Scalar(0));
  vector<Mat> r5, t5;

  const int flags = omnidir::CALIB_FIX_SKEW;
  TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-7);

  Mat idx4, idx5;

  double rms4 = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize, K4,
                                       xi4, D4, r4, t4, flags, criteria, idx4);

  double rms5 = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize, K5,
                                       xi5, D5, r5, t5, flags, criteria, idx5);

  // Ideally, both runs should use the same number of views; if not, reproj RMS
  // still computed correctly by idx.
  EXPECT_GT((int)idx4.total(), 0);
  EXPECT_GT((int)idx5.total(), 0);



  // Expect 5-param to fit at least as well as 4-param (on their internal
  // objective)
  EXPECT_LE(rms5, rms4);

  // Recompute RMS with projectPoints (xi is double there), ONLY on used views
  // (idx)
  const double xi4v = xi4.at<double>(0, 0);
  const double xi5v = xi5.at<double>(0, 0);

  const double rms4_check =
      computeReprojRms(objectPoints, imagePoints, K4, xi4v, D4, r4, t4, idx4);
  const double rms5_check =
      computeReprojRms(objectPoints, imagePoints, K5, xi5v, D5, r5, t5, idx5);

  // These can differ slightly from internal calibrate RMS; keep a reasonable
  // tolerance.
  EXPECT_LE(rms5_check, rms4_check);


  ASSERT_EQ((int)D5.total(), 5);
  const double k3_est = D5.reshape(1, 1).at<double>(0, 4);

  EXPECT_TRUE(std::isfinite(k3_est));
  EXPECT_GT(k3_est, -1e3);
  EXPECT_LT(k3_est, 1e3);
}

// Verifies backward compatibility of undistortPoints: 4 coefficients
// produce identical results to 5 coefficients with k3 equal to zero.

TEST(Omnidir_K3,UndistortPoints_backward_compat_4coeffs_equals_5coeffs_k3_zero) {
  using namespace cv;

  const Matx33d K(520, 0, 320, 0, 520, 240, 0, 0, 1);

  Mat xiMat(1, 1, CV_64F);
  xiMat.at<double>(0, 0) = 1.2;

  // D4
  Mat D4(1, 4, CV_64F);
  D4.at<double>(0, 0) = -0.10;
  D4.at<double>(0, 1) = 0.01;
  D4.at<double>(0, 2) = 0.001;
  D4.at<double>(0, 3) = -0.0005;

  // D5 with k3=0
  Mat D5(1, 5, CV_64F);
  D5.at<double>(0, 0) = -0.10;
  D5.at<double>(0, 1) = 0.01;
  D5.at<double>(0, 2) = 0.001;
  D5.at<double>(0, 3) = -0.0005;
  D5.at<double>(0, 4) = 0.0;

  // create some distorted pixels by projecting points with D5(k3=0)
  vector<Point3f> obj = makeChessboard3D(Size(9, 6), 0.04f);
  Mat rvec = (Mat_<double>(3, 1) << 0.2, -0.1, 0.05);
  Mat tvec = (Mat_<double>(3, 1) << 0.40, -0.25, 1.80);

  vector<Point2f> distorted;
  omnidir::projectPoints(obj, distorted, rvec, tvec, K, xiMat.at<double>(0, 0),
                         D5);

  Mat und4, und5;
  omnidir::undistortPoints(distorted, und4, K, D4, xiMat, Matx33d::eye());
  omnidir::undistortPoints(distorted, und5, K, D5, xiMat, Matx33d::eye());

  ASSERT_FALSE(und4.empty());
  ASSERT_FALSE(und5.empty());
  ASSERT_EQ(und4.total(), und5.total());
  ASSERT_EQ(und4.type(), und5.type());

  // undistortPoints returns Nx1 2-channel (or 1xN). Compare numerically.
  const double maxAbs = maxAbsDiff(und4, und5);
  EXPECT_LT(maxAbs, 1e-8);
}

// Confirms that undistortPoints output changes when only the k3
// distortion coefficient is modified.

TEST(Omnidir_K3, UndistortPoints_changes_when_only_k3_changes) {
  using namespace cv;

  const Matx33d K(520, 0, 320, 0, 520, 240, 0, 0, 1);

  Mat xiMat(1, 1, CV_64F);
  xiMat.at<double>(0, 0) = 1.2;

  Mat Dk3_0(1, 5, CV_64F);
  Dk3_0.at<double>(0, 0) = -0.10;
  Dk3_0.at<double>(0, 1) = 0.01;
  Dk3_0.at<double>(0, 2) = 0.001;
  Dk3_0.at<double>(0, 3) = -0.0005;
  Dk3_0.at<double>(0, 4) = 0.0;

  Mat Dk3_1 = Dk3_0.clone();
  Dk3_1.at<double>(0, 4) = -0.2; // strong k3 change to avoid numerical ~0

  // Make distorted pixels from the k3!=0 model
  vector<Point3f> obj = makeChessboard3D(Size(9, 6), 0.04f);
  Mat rvec = (Mat_<double>(3, 1) << 0.2, -0.1, 0.05);
  Mat tvec = (Mat_<double>(3, 1) << 0.45, -0.30, 1.80);

  vector<Point2f> distorted;
  omnidir::projectPoints(obj, distorted, rvec, tvec, K, xiMat.at<double>(0, 0),
                         Dk3_1);

  // Undistort the SAME distorted pixels with different k3 values.
  Mat und0, und1;
  omnidir::undistortPoints(distorted, und0, K, Dk3_0, xiMat, Matx33d::eye());
  omnidir::undistortPoints(distorted, und1, K, Dk3_1, xiMat, Matx33d::eye());

  ASSERT_FALSE(und0.empty());
  ASSERT_FALSE(und1.empty());
  ASSERT_EQ(und0.total(), und1.total());
  ASSERT_EQ(und0.type(), und1.type());

  // If k3 is used inside undistort, these should differ measurably.
  const double maxAbs = maxAbsDiff(und0, und1);
  EXPECT_GT(maxAbs, 1e-6);

  // sanity: finite values
  Mat und1_flat = und1.reshape(1);
  for (int i = 0; i < und1_flat.rows; ++i) {
    const double v = und1_flat.at<double>(i, 0);
    EXPECT_TRUE(std::isfinite(v));
  }
}

// Ensures rectify maps are identical for 4 coefficients and 5 coefficients
// with k3=0, and differ when k3 is non-zero.

TEST(
    Omnidir_K3,
    InitUndistortRectifyMap_maps_equal_for_k3_zero_and_change_when_k3_changes) {
  using namespace cv;

  const Size sz(640, 480);
  const Matx33d K(520, 0, 320, 0, 520, 240, 0, 0, 1);

  Mat xiMat(1, 1, CV_64F);
  xiMat.at<double>(0, 0) = 1.2;

  Mat D4(1, 4, CV_64F);
  D4.at<double>(0, 0) = -0.10;
  D4.at<double>(0, 1) = 0.01;
  D4.at<double>(0, 2) = 0.001;
  D4.at<double>(0, 3) = -0.0005;

  Mat D5_k3_0(1, 5, CV_64F);
  D5_k3_0.at<double>(0, 0) = -0.10;
  D5_k3_0.at<double>(0, 1) = 0.01;
  D5_k3_0.at<double>(0, 2) = 0.001;
  D5_k3_0.at<double>(0, 3) = -0.0005;
  D5_k3_0.at<double>(0, 4) = 0.0;

  Mat D5_k3_1 = D5_k3_0.clone();
  D5_k3_1.at<double>(0, 4) = -0.2;

  const int flags = omnidir::RECTIFY_PERSPECTIVE;

  Mat m1_4, m2_4, m1_5z, m2_5z, m1_5k3, m2_5k3;

  omnidir::initUndistortRectifyMap(K, D4, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, m1_4, m2_4, flags);
  omnidir::initUndistortRectifyMap(K, D5_k3_0, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, m1_5z, m2_5z, flags);
  omnidir::initUndistortRectifyMap(K, D5_k3_1, xiMat, Matx33d::eye(), K, sz,
                                   CV_32FC1, m1_5k3, m2_5k3, flags);

  ASSERT_FALSE(m1_4.empty());
  ASSERT_FALSE(m2_4.empty());
  ASSERT_FALSE(m1_5z.empty());
  ASSERT_FALSE(m2_5z.empty());
  ASSERT_FALSE(m1_5k3.empty());
  ASSERT_FALSE(m2_5k3.empty());

  // 4 coeffs must match 5 coeffs with k3=0
  EXPECT_LT(maxAbsDiff(m1_4, m1_5z), 1e-6);
  EXPECT_LT(maxAbsDiff(m2_4, m2_5z), 1e-6);

  // changing k3 should change maps
  EXPECT_GT(maxAbsDiff(m1_5z, m1_5k3), 1e-5);
  EXPECT_GT(maxAbsDiff(m2_5z, m2_5k3), 1e-5);
}

} // namespace opencv_test
