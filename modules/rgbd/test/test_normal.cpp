/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdexcept>

#include <opencv2/contrib.hpp>
#include <opencv2/rgbd.hpp>

#include "test_precomp.hpp"

cv::Point3f
rayPlaneIntersection(cv::Point2f uv, const cv::Mat& centroid, const cv::Mat& normal, const cv::Mat_<float>& Kinv);

cv::Vec3f
rayPlaneIntersection(const cv::Vec3d& uv1, double centroid_dot_normal, const cv::Vec3d& normal,
                     const cv::Matx33d& Kinv);
cv::Vec3f
rayPlaneIntersection(const cv::Vec3d& uv1, double centroid_dot_normal, const cv::Vec3d& normal, const cv::Matx33d& Kinv)
{

  cv::Matx31d L = Kinv * uv1; //a ray passing through camera optical center
  //and uv.
  L = L * (1.0 / cv::norm(L));
  double LdotNormal = L.dot(normal);
  double d;
  if (std::fabs(LdotNormal) > 1e-9)
  {
    d = centroid_dot_normal / LdotNormal;
  }
  else
  {
    d = 1.0;
    std::cout << "warning, LdotNormal nearly 0! " << LdotNormal << std::endl;
    std::cout << "contents of L, Normal: " << cv::Mat(L) << ", " << cv::Mat(normal) << std::endl;
  }
  cv::Vec3f xyz((float)(d * L(0)), (float)(d * L(1)), (float)(d * L(2)));
  return xyz;
}

cv::Point3f
rayPlaneIntersection(cv::Point2f uv, const cv::Mat& centroid, const cv::Mat& normal, const cv::Mat_<float>& Kinv)
{
  cv::Matx33d dKinv(Kinv);
  cv::Vec3d dNormal(normal);
  return rayPlaneIntersection(cv::Vec3d(uv.x, uv.y, 1), centroid.dot(normal), dNormal, dKinv);
}

const int W = 640;
const int H = 480;
int window_size = 5;
float focal_length = 525;
float cx = W / 2.f + 0.5f;
float cy = H / 2.f + 0.5f;

cv::Mat K = (cv::Mat_<double>(3, 3) << focal_length, 0, cx, 0, focal_length, cy, 0, 0, 1);
cv::Mat Kinv = K.inv();

static cv::RNG rng;
struct Plane
{

  cv::Vec3d n, p;
  double p_dot_n;
  Plane()
  {
    n[0] = rng.uniform(-0.5, 0.5);
    n[1] = rng.uniform(-0.5, 0.5);
    n[2] = -0.3; //rng.uniform(-1.f, 0.5f);
    n = n / cv::norm(n);
    set_d((float)rng.uniform(-2.0, 0.6));
  }

  void
  set_d(float d)
  {
    p = cv::Vec3d(0, 0, d / n[2]);
    p_dot_n = p.dot(n);
  }

  cv::Vec3f
  intersection(float u, float v, const cv::Matx33f& Kinv_in) const
  {
    return rayPlaneIntersection(cv::Vec3d(u, v, 1), p_dot_n, n, Kinv_in);
  }
};

void
gen_points_3d(std::vector<Plane>& planes_out, cv::Mat_<unsigned char> &plane_mask, cv::Mat& points3d, cv::Mat& normals,
              int n_planes);
void
gen_points_3d(std::vector<Plane>& planes_out, cv::Mat_<unsigned char> &plane_mask, cv::Mat& points3d, cv::Mat& normals,
              int n_planes)
{
  std::vector<Plane> planes;
  for (int i = 0; i < n_planes; i++)
  {
    Plane px;
    for (int j = 0; j < 1; j++)
    {
      px.set_d(rng.uniform(-3.f, -0.5f));
      planes.push_back(px);
    }
  }
  cv::Mat_ < cv::Vec3f > outp(H, W);
  cv::Mat_ < cv::Vec3f > outn(H, W);
  plane_mask.create(H, W);

  // n  ( r - r_0) = 0
  // n * r_0 = d
  //
  // r_0 = (0,0,0)
  // r[0]
  for (int v = 0; v < H; v++)
  {
    for (int u = 0; u < W; u++)
    {
      unsigned int plane_index = (unsigned int)((u / float(W)) * planes.size());
      Plane plane = planes[plane_index];
      outp(v, u) = plane.intersection((float)u, (float)v, Kinv);
      outn(v, u) = plane.n;
      plane_mask(v, u) = (uchar)plane_index;
    }
  }
  planes_out = planes;
  points3d = outp;
  normals = outn;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_RgbdNormalsTest: public cvtest::BaseTest
{
public:
  CV_RgbdNormalsTest()
  {
  }
  ~CV_RgbdNormalsTest()
  {
  }
protected:
  void
  run(int)
  {
    try
    {
      cv::Mat_<unsigned char> plane_mask;
      for (unsigned char i = 0; i < 3; ++i)
      {
        cv::RgbdNormals::RGBD_NORMALS_METHOD method;
        // inner vector: whether it's 1 plane or 3 planes
        // outer vector: float or double
        std::vector<std::vector<float> > errors(2);
        errors[0].resize(2);
        errors[1].resize(2);
        switch (i)
        {
          case 0:
            method = cv::RgbdNormals::RGBD_NORMALS_METHOD_FALS;
            std::cout << std::endl << "*** FALS" << std::endl;
            errors[0][0] = 0.006f;
            errors[0][1] = 0.03f;
            errors[1][0] = 0.00008f;
            errors[1][1] = 0.02f;
            break;
          case 1:
            method = cv::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD;
            std::cout << std::endl << "*** LINEMOD" << std::endl;
            errors[0][0] = 0.04f;
            errors[0][1] = 0.07f;
            errors[1][0] = 0.05f;
            errors[1][1] = 0.08f;
            break;
          case 2:
            method = cv::RgbdNormals::RGBD_NORMALS_METHOD_SRI;
            std::cout << std::endl << "*** SRI" << std::endl;
            errors[0][0] = 0.02f;
            errors[0][1] = 0.04f;
            errors[1][0] = 0.02f;
            errors[1][1] = 0.04f;
            break;
		  default:
			method = (cv::RgbdNormals::RGBD_NORMALS_METHOD)-1;
			CV_Error(0, "");
        }

        for (unsigned char j = 0; j < 2; ++j)
        {
          int depth = (j % 2 == 0) ? CV_32F : CV_64F;
          if (depth == CV_32F)
            std::cout << "* float" << std::endl;
          else
            std::cout << "* double" << std::endl;

          cv::RgbdNormals normals_computer(H, W, depth, K, 5, method);
          normals_computer.initialize();

          std::vector<Plane> plane_params;
          cv::Mat points3d, ground_normals;
          // 1 plane, continuous scene, very low error..
          std::cout << "1 plane" << std::endl;
          float err_mean = 0;
          for (int ii = 0; ii < 5; ++ii)
          {
            gen_points_3d(plane_params, plane_mask, points3d, ground_normals, 1);
            err_mean += testit(points3d, ground_normals, normals_computer);
          }
          std::cout << "mean diff: " << (err_mean / 5) << std::endl;
          EXPECT_LE(err_mean/5, errors[j][0])<< " thresh: " << errors[j][0] << std::endl;

          // 3 discontinuities, more error expected.
          std::cout << "3 planes" << std::endl;
          err_mean = 0;
          for (int ii = 0; ii < 5; ++ii)
          {
            gen_points_3d(plane_params, plane_mask, points3d, ground_normals, 3);
            err_mean += testit(points3d, ground_normals, normals_computer);
          }
          std::cout << "mean diff: " << (err_mean / 5) << std::endl;
          EXPECT_LE(err_mean/5, errors[j][1])<< "mean diff: " << (err_mean/5) << " thresh: " << errors[j][1] << std::endl;
        }
      }

      //TODO test NaNs in data

    } catch (...)
    {
      ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    }
    ts->set_failed_test_info(cvtest::TS::OK);
  }

  float
  testit(const cv::Mat & points3d, const cv::Mat & in_ground_normals, const cv::RgbdNormals & normals_computer)
  {
    cv::TickMeter tm;
    tm.start();
    cv::Mat in_normals;
    if (normals_computer.method() == cv::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD)
    {
      std::vector<cv::Mat> channels;
      cv::split(points3d, channels);
      normals_computer(channels[2], in_normals);
    }
    else
      normals_computer(points3d, in_normals);
    tm.stop();

    cv::Mat_<cv::Vec3f> normals, ground_normals;
    in_normals.convertTo(normals, CV_32FC3);
    in_ground_normals.convertTo(ground_normals, CV_32FC3);

    float err = 0;
    for (int y = 0; y < normals.rows; ++y)
      for (int x = 0; x < normals.cols; ++x)
      {
        cv::Vec3f vec1 = normals(y, x), vec2 = ground_normals(y, x);
        vec1 = vec1 / cv::norm(vec1);
        vec2 = vec2 / cv::norm(vec2);

        float dot = vec1.dot(vec2);
        // Just for rounding errors
        if (std::abs(dot) < 1)
          err += std::min(std::acos(dot), std::acos(-dot));
      }

    err /= normals.rows * normals.cols;
    std::cout << "Average error: " << err << " Speed: " << tm.getTimeMilli() << " ms" << std::endl;
    return err;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_RgbdPlaneTest: public cvtest::BaseTest
{
public:
  CV_RgbdPlaneTest()
  {
  }
  ~CV_RgbdPlaneTest()
  {
  }
protected:
  void
  run(int)
  {
    try
    {
      cv::RgbdPlane plane_computer;

      std::vector<Plane> planes;
      cv::Mat points3d, ground_normals;
      cv::Mat_<unsigned char> plane_mask;
      gen_points_3d(planes, plane_mask, points3d, ground_normals, 1);
      testit(planes, plane_mask, points3d, plane_computer); // 1 plane, continuous scene, very low error..
      for (int ii = 0; ii < 10; ii++)
      {
        gen_points_3d(planes, plane_mask, points3d, ground_normals, 3); //three planes
        testit(planes, plane_mask, points3d, plane_computer); // 3 discontinuities, more error expected.
      }
    } catch (...)
    {
      ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    }
    ts->set_failed_test_info(cvtest::TS::OK);
  }

  void
  testit(const std::vector<Plane> & gt_planes, const cv::Mat & gt_plane_mask, const cv::Mat & points3d,
         cv::RgbdPlane & plane_computer)
  {
    for (char i_test = 0; i_test < 2; ++i_test)
    {
      cv::TickMeter tm1, tm2;
      cv::Mat plane_mask;
      std::vector<cv::Vec4f> plane_coefficients;

      if (i_test == 0)
      {
        tm1.start();
        // First, get the normals
        int depth = CV_32F;
        cv::RgbdNormals normals_computer(H, W, depth, K, 5, cv::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
        cv::Mat normals;
        normals_computer(points3d, normals);
        tm1.stop();

        tm2.start();
        plane_computer(points3d, normals, plane_mask, plane_coefficients);
        tm2.stop();
      }
      else
      {
        tm2.start();
        plane_computer(points3d, plane_mask, plane_coefficients);
        tm2.stop();
      }

      // Compare each found plane to each ground truth plane
      int n_planes = (int)plane_coefficients.size();
      int n_gt_planes = (int)gt_planes.size();
      cv::Mat_<int> matching(n_gt_planes, n_planes);
      for (int j = 0; j < n_gt_planes; ++j)
      {
        cv::Mat gt_mask = gt_plane_mask == j;
        int n_gt = cv::countNonZero(gt_mask);
        int n_max = 0, i_max = 0;
        for (int i = 0; i < n_planes; ++i)
        {
          cv::Mat dst;
          cv::bitwise_and(gt_mask, plane_mask == i, dst);
          matching(j, i) = cv::countNonZero(dst);
          if (matching(j, i) > n_max)
          {
            n_max = matching(j, i);
            i_max = i;
          }
        }
        // Get the best match
        ASSERT_LE(float(n_max - n_gt) / n_gt, 0.001);
        // Compare the normals
        cv::Vec3d normal(plane_coefficients[i_max][0], plane_coefficients[i_max][1], plane_coefficients[i_max][2]);
        ASSERT_GE(std::abs(gt_planes[j].n.dot(normal)), 0.95);
      }

      std::cout << " Speed: ";
      if (i_test == 0)
        std::cout << "normals " << tm1.getTimeMilli() << " ms and ";
      std::cout << "plane " << tm2.getTimeMilli() << " ms " << std::endl;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Rgbd_Normals, compute)
{
  CV_RgbdNormalsTest test;
  test.safe_run();
}

TEST(Rgbd_Plane, compute)
{
  CV_RgbdPlaneTest test;
  test.safe_run();
}
