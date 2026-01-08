// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test {

void generateTwoViewRandomScene( TwoViewDataSet &data )
{
    vector<Mat_<double> > points2d;
    vector<cv::Matx33d> Rs;
    vector<cv::Vec3d> ts;
    vector<cv::Matx34d> Ps;
    Matx33d K;
    Mat_<double> points3d;

    int nviews = 2;
    int npoints = 30;
    bool is_projective = true;

    generateScene(nviews, npoints, is_projective, K, Rs, ts, Ps, points3d, points2d);

    // Internal parameters (same K)
    data.K1 = K;
    data.K2 = K;

    // Rotation
    data.R1 = Rs[0];
    data.R2 = Rs[1];

    // Translation
    data.t1 = ts[0];
    data.t2 = ts[1];

    // Projection matrix, P = K(R|t)
    data.P1 = Ps[0];
    data.P2 = Ps[1];

    // Fundamental matrix
    fundamentalFromProjections( data.P1, data.P2, data.F );

    // 3D points
    data.X = points3d;

    // Projected points
    data.x1 = points2d[0];
    data.x2 = points2d[1];
}

/** Check the properties of a fundamental matrix:
*
*   1. The determinant is 0 (rank deficient)
*   2. The condition x'T*F*x = 0 is satisfied to precision.
*/
void
expectFundamentalProperties( const cv::Matx33d &F,
                             const cv::Mat_<double> &ptsA,
                             const cv::Mat_<double> &ptsB,
                             double precision )
{
  EXPECT_NEAR( 0, determinant(F), precision );

  int n = ptsA.cols;
  EXPECT_EQ( n, ptsB.cols );

  cv::Mat_<double> x1, x2;
  euclideanToHomogeneous( ptsA, x1 );
  euclideanToHomogeneous( ptsB, x2 );

  for( int i = 0; i < n; ++i )
  {
    double residual = Vec3d(x2(0,i),x2(1,i),x2(2,i)).ddot( F * Vec3d(x1(0,i),x1(1,i),x1(2,i)) );
    EXPECT_NEAR( 0.0, residual, precision );
  }
}


void
parser_2D_tracks(const string &_filename, std::vector<Mat> &points2d )
{
  std::ifstream myfile(_filename.c_str());

  if (!myfile.is_open())
      CV_Error(cv::Error::StsError, string("Unable to read file: ") + _filename + "\n");
  else {

    double x, y;
    string line_str;
    Mat nan_mat = Mat(2, 1 , CV_64F, -1);
    int n_frames = 0, n_tracks = 0, track = 0;

    while ( getline(myfile, line_str) )
    {
      std::istringstream line(line_str);

      if ( track > n_tracks )
      {
        n_tracks = track;

        for (int i = 0; i < n_frames; ++i)
          cv::hconcat(points2d[i], nan_mat, points2d[i]);
      }

      for (int frame = 1; line >> x >> y; ++frame)
      {
        if ( frame > n_frames )
        {
          n_frames = frame;
          points2d.push_back(nan_mat);
        }

        points2d[frame-1].at<double>(0,track) = x;
        points2d[frame-1].at<double>(1,track) = y;
      }

      ++track;
    }

    myfile.close();
  }

}

} // namespace
