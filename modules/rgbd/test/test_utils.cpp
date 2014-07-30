#include "test_precomp.hpp"

#include <opencv2/calib3d.hpp>

namespace cv
{
namespace rgbd
{

class CV_RgbdDepthTo3dTest: public cvtest::BaseTest
{
public:
  CV_RgbdDepthTo3dTest()
  {
  }
  ~CV_RgbdDepthTo3dTest()
  {
  }
protected:
  void
  run(int)
  {
    try
    {
      // K from a VGA Kinect
      Mat K = (Mat_<float>(3, 3) << 525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);

      // Create a random depth image
      RNG rng;
      Mat_<float> depth(480, 640);
      rng.fill(depth, RNG::UNIFORM, 0, 100);

      // Create some 3d points on the plane
      int rows = depth.rows, cols = depth.cols;
      Mat_<Vec3f> points3d;
      depthTo3d(depth, K, points3d);

      // Make sure the points belong to the plane
      Mat points = points3d.reshape(1, rows*cols);
      Mat image_points;
      Mat rvec;
      Rodrigues(Mat::eye(3,3,CV_32F),rvec);
      Mat tvec = (Mat_<float>(1,3) << 0, 0, 0);
      projectPoints(points, rvec, tvec, K, Mat(), image_points);
      image_points = image_points.reshape(2, rows);

      float avg_diff = 0;
      for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
          avg_diff += (float)norm(image_points.at<Vec2f>(y,x) - Vec2f((float)x,(float)y));

      // Verify the function works
      ASSERT_LE(avg_diff/rows/cols, 1e-4) << "Average error for ground truth is: " << (avg_diff / rows / cols);
    } catch (...)
    {
      ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    }
    ts->set_failed_test_info(cvtest::TS::OK);
  }
};

}
}

TEST(Rgbd_DepthTo3d, compute)
{
  cv::rgbd::CV_RgbdDepthTo3dTest test;
  test.safe_run();
}
