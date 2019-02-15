// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <opencv2/sfm.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ts.hpp>
#include <opencv2/core.hpp>

#include "scene.h"

#define OPEN_TESTFILE(FNAME,FS)  \
      FS.open(FNAME, FileStorage::READ); \
    if (!FS.isOpened())\
    {\
        std::cerr << "Cannot find file: " << FNAME << std::endl;\
        return;\
    }

namespace opencv_test
{
  using namespace cv::sfm;

  template<typename T>
  inline void
  EXPECT_MATRIX_NEAR(const T a, const T b, double tolerance)
  {
    bool dims_match = (a.rows == b.rows) && (a.cols == b.cols);
    EXPECT_EQ((int)a.rows, (int)b.rows);
    EXPECT_EQ((int)a.cols, (int)b.cols);

    if (dims_match)
    {
      for (int r = 0; r < a.rows; ++r)
      {
        for (int c = 0; c < a.cols; ++c)
        {
          EXPECT_NEAR(a(r, c), b(r, c), tolerance) << "r=" << r << ", c=" << c << ".";
        }
      }
    }
  }

  template<typename T>
  inline void
  EXPECT_VECTOR_NEAR(const T a, const T b, double tolerance)
  {
    bool dims_match = (a.rows == b.rows);
    EXPECT_EQ((int)a.rows,(int)b.rows) << "Matrix rows don't match.";

    if (dims_match)
    {
      for (int r = 0; r < a.rows; ++r)
      {
        EXPECT_NEAR(a(r), b(r), tolerance) << "r=" << r << ".";
      }
    }
  }

  template<class T>
  inline double
  cosinusBetweenMatrices(const T &a, const T &b)
  {
    double s = cv::sum( a.mul(b) )[0];
    return ( s / cv::norm(a) / cv::norm(b) );
  }

  // Check that sin(angle(a, b)) < tolerance
  template<typename T>
  inline void
  EXPECT_MATRIX_PROP(const T a, const T b, double tolerance)
  {
    bool dims_match = (a.rows == b.rows) && (a.cols == b.cols);
    EXPECT_EQ((int)a.rows, (int)b.rows);
    EXPECT_EQ((int)a.cols, (int)b.cols);

    if (dims_match)
    {
      double c = cosinusBetweenMatrices(a, b);
      if (c * c < 1)
      {
        double s = sqrt(1 - c * c);
        EXPECT_NEAR(0, s, tolerance);
      }
    }
  }




  struct TwoViewDataSet
  {
    cv::Matx33d K1, K2; // Internal parameters
    cv::Matx33d R1, R2; // Rotation
    cv::Vec3d t1, t2; // Translation
    cv::Matx34d P1, P2; // Projection matrix, P = K(R|t)
    cv::Matx33d F; // Fundamental matrix
    cv::Mat_<double> X; // 3D points
    cv::Mat_<double> x1, x2; // Projected points
  };

  void
  generateTwoViewRandomScene(TwoViewDataSet &data);

  /** Check the properties of a fundamental matrix:
  *
  *   1. The determinant is 0 (rank deficient)
  *   2. The condition x'T*F*x = 0 is satisfied to precision.
  */
  void
  expectFundamentalProperties( const cv::Matx33d &F,
                               const cv::Mat_<double> &ptsA,
                               const cv::Mat_<double> &ptsB,
                               double precision = 1e-9 );

  /**
   * 2D tracked points
   * -----------------
   *
   * The format is:
   *
   * row1 : x1 y1 x2 y2 ... x36 y36 for track 1
   * row2 : x1 y1 x2 y2 ... x36 y36 for track 2
   * etc
   *
   * i.e. a row gives the 2D measured position of a point as it is tracked
   * through frames 1 to 36.  If there is no match found in a view then x
   * and y are -1.
   *
   * Each row corresponds to a different point.
   *
   */
  void
  parser_2D_tracks(const std::string &_filename, std::vector<cv::Mat> &points2d );

} // namespace

#endif
