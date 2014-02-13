#include <opencv2/calib3d.hpp>

#include "test_precomp.hpp"

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
      cv::Mat K = (cv::Mat_<float>(3, 3) << 525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);

      // Create a random depth image
      cv::RNG rng;
      cv::Mat_<float> depth(480, 640);
      rng.fill(depth, cv::RNG::UNIFORM, 0, 100);

      // Create some 3d points on the plane
      int rows = depth.rows, cols = depth.cols;
      cv::Mat_<cv::Vec3f> points3d;
      cv::depthTo3d(depth, K, points3d);

      // Make sure the points belong to the plane
      cv::Mat points = points3d.reshape(1, rows*cols);
      cv::Mat image_points;
      cv::Mat rvec;
      cv::Rodrigues(cv::Mat::eye(3,3,CV_32F),rvec);
      cv::Mat tvec = (cv::Mat_<float>(1,3) << 0, 0, 0);
      cv::projectPoints(points, rvec, tvec, K, cv::Mat(), image_points);
      image_points = image_points.reshape(2, rows);

      float avg_diff = 0;
      for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
          avg_diff += cv::norm(image_points.at<cv::Vec2f>(y,x) - cv::Vec2f(x,y));

      // Verify the function works
      ASSERT_LE(avg_diff/rows/cols, 1e-4) << "Average error for ground truth is: " << (avg_diff / rows / cols);
    } catch (...)
    {
      ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    }
    ts->set_failed_test_info(cvtest::TS::OK);
  }
};

TEST(Rgbd_DepthTo3d, compute)
{
  CV_RgbdDepthTo3dTest test;
  test.safe_run();
}
