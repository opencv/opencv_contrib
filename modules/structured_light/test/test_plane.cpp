/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "test_precomp.hpp"
#include <opencv2/rgbd.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const string STRUCTURED_LIGHT_DIR = "structured_light";
const string FOLDER_DATA = "data";

/****************************************************************************************\
*                                    Plane test                                          *
 \****************************************************************************************/
class CV_PlaneTest : public cvtest::BaseTest
{
 public:
  CV_PlaneTest();
  ~CV_PlaneTest();

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // From rgbd module: since I needed the distance method of plane class, I copied the class from rgb module
  // it will be made a pull request to make Plane class public

  /** Structure defining a plane. The notations are from the second paper */
  class PlaneBase
  {
   public:
    PlaneBase(const Vec3f & m, const Vec3f &n_in, int index) :
        index_(index),
        n_(n_in),
        m_sum_(Vec3f(0, 0, 0)),
        m_(m),
        Q_(Matx33f::zeros()),
        mse_(0),
        K_(0)
    {
      UpdateD();
    }

    virtual ~PlaneBase()
    {
    }

    /** Compute the distance to the plane. This will be implemented by the children to take into account different
     * sensor models
     * @param p_j
     * @return
     */
    virtual
    float
    distance(const Vec3f& p_j) const = 0;

    /** The d coefficient in the plane equation ax+by+cz+d = 0
     * @return
     */
    inline float d() const
    {
      return d_;
    }

    /** The normal to the plane
     * @return the normal to the plane
     */
    const Vec3f &
    n() const
    {
      return n_;
    }

    /** Update the different coefficients of the plane, based on the new statistics
     */
    void UpdateParameters()
    {
      if( empty() )
        return;
      m_ = m_sum_ / K_;
      // Compute C
      Matx33f C = Q_ - m_sum_ * m_.t();

      // Compute n
      SVD svd(C);
      n_ = Vec3f(svd.vt.at<float>(2, 0), svd.vt.at<float>(2, 1), svd.vt.at<float>(2, 2));
      mse_ = svd.w.at<float>(2) / K_;

      UpdateD();
    }

    /** Update the different sum of point and sum of point*point.t()
     */
    void UpdateStatistics(const Vec3f & point, const Matx33f & Q_local)
    {
      m_sum_ += point;
      Q_ += Q_local;
      ++K_;
    }

    inline size_t empty() const
    {
      return K_ == 0;
    }

    inline int K() const
    {
      return K_;
    }
    /** The index of the plane */
    int index_;
   protected:
    /** The 4th coefficient in the plane equation ax+by+cz+d = 0 */
    float d_;
    /** Normal of the plane */
    Vec3f n_;
   private:
    inline void UpdateD()
    {
      // Hessian form (d = nc . p_plane (centroid here) + p)
      //d = -1 * n.dot (xyz_centroid);//d =-axP+byP+czP
      d_ = -m_.dot(n_);
    }
    /** The sum of the points */
    Vec3f m_sum_;
    /** The mean of the points */
    Vec3f m_;
    /** The sum of pi * pi^\top */
    Matx33f Q_;
    /** The different matrices we need to update */
    Matx33f C_;
    float mse_;
    /** the number of points that form the plane */
    int K_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Basic planar child, with no sensor error model
   */
  class Plane : public PlaneBase
  {
   public:
    Plane(const Vec3f & m, const Vec3f &n_in, int index) :
        PlaneBase(m, n_in, index)
    {
    }

    /** The computed distance is perfect in that case
     * @param p_j the point to compute its distance to
     * @return
     */
    float distance(const Vec3f& p_j) const
    {
      return std::abs(float(p_j.dot(n_) + d_));
    }
  };
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 protected:
  void run( int );

};

CV_PlaneTest::CV_PlaneTest(){}

CV_PlaneTest::~CV_PlaneTest(){}

void CV_PlaneTest::run( int )
{
  string folder = cvtest::TS::ptr()->get_data_path() + "/" + STRUCTURED_LIGHT_DIR + "/" + FOLDER_DATA + "/";
  structured_light::GrayCodePattern::Params params;
  params.width = 1280;
  params.height = 800;
  // Set up GraycodePattern with params
  Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create( params );
  size_t numberOfPatternImages = graycode->getNumberOfPatternImages();


  FileStorage fs( folder + "calibrationParameters.yml", FileStorage::READ );
  if( !fs.isOpened() )
  {
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
  }

  FileStorage fs2( folder + "gt_plane.yml", FileStorage::READ );
  if( !fs.isOpened() )
  {
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
  }

  // Loading ground truth plane parameters
  Vec4f plane_coefficients;
  Vec3f m;
  fs2["plane_coefficients"] >> plane_coefficients;
  fs2["m"] >> m;

  // Loading calibration parameters
  Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;

  fs["cam1_intrinsics"] >> cam1intrinsics;
  fs["cam2_intrinsics"] >> cam2intrinsics;
  fs["cam1_distorsion"] >> cam1distCoeffs;
  fs["cam2_distorsion"] >> cam2distCoeffs;
  fs["R"] >> R;
  fs["T"] >> T;

  // Loading white and black images
  vector<Mat> blackImages;
  vector<Mat> whiteImages;

  blackImages.resize( 2 );
  whiteImages.resize( 2 );

  whiteImages[0] = imread( folder + "pattern_cam1_im43.jpg", 0 );
  whiteImages[1] = imread( folder + "pattern_cam2_im43.jpg", 0 );
  blackImages[0] = imread( folder + "pattern_cam1_im44.jpg", 0 );
  blackImages[1] = imread( folder + "pattern_cam2_im44.jpg", 0 );

  Size imagesSize = whiteImages[0].size();

  if( ( !cam1intrinsics.data ) || ( !cam2intrinsics.data ) || ( !cam1distCoeffs.data ) || ( !cam2distCoeffs.data ) || ( !R.data )
      || ( !T.data ) || ( !whiteImages[0].data ) || ( !whiteImages[1].data ) || ( !blackImages[0].data )
      || ( !blackImages[1].data ) )
  {
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
  }

  // Computing stereo rectify parameters
  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];
  stereoRectify( cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, 0,
                 -1, imagesSize, &validRoi[0], &validRoi[1] );

  Mat map1x, map1y, map2x, map2y;
  initUndistortRectifyMap( cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y );
  initUndistortRectifyMap( cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y );

  vector<vector<Mat> > captured_pattern;
  captured_pattern.resize( 2 );
  captured_pattern[0].resize( numberOfPatternImages );
  captured_pattern[1].resize( numberOfPatternImages );

  // Loading and rectifying pattern images
  for( size_t i = 0; i < numberOfPatternImages; i++ )
  {
    ostringstream name1;
    name1 << "pattern_cam1_im" << i + 1 << ".jpg";
    captured_pattern[0][i] = imread( folder + name1.str(), 0 );
    ostringstream name2;
    name2 << "pattern_cam2_im" << i + 1 << ".jpg";
    captured_pattern[1][i] = imread( folder + name2.str(), 0 );

    if( (!captured_pattern[0][i].data) || (!captured_pattern[1][i].data) )
    {
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
    }

    remap( captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    remap( captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  }

  // Rectifying white and black images
  remap( whiteImages[0], whiteImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( whiteImages[1], whiteImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );

  remap( blackImages[0], blackImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( blackImages[1], blackImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );

  // Setting up threshold parameters to reconstruct only the plane in foreground
  graycode->setBlackThreshold( 55 );
  graycode->setWhiteThreshold( 10 );

  // Computing the disparity map
  Mat disparityMap;
  bool decoded = graycode->decode( captured_pattern, disparityMap, blackImages, whiteImages,
                                   structured_light::DECODE_3D_UNDERWORLD );
  EXPECT_TRUE( decoded );

  // Computing the point cloud
  Mat pointcloud;
  disparityMap.convertTo( disparityMap, CV_32FC1 );
  reprojectImageTo3D( disparityMap, pointcloud, Q, true, -1 );
  // from mm (unit of calibration) to m
  pointcloud = pointcloud / 1000;

  // Setting up plane with ground truth plane values
  Vec3f normal( plane_coefficients.val[0], plane_coefficients.val[1], plane_coefficients.val[2] );
  Ptr<PlaneBase> plane = Ptr<PlaneBase>( new Plane( m, normal, 0 ) );

  // Computing the distance of every point of the pointcloud from ground truth plane
  float sum_d = 0;
  int cont = 0;
  for( int i = 0; i < disparityMap.rows; i++ )
  {
    for( int j = 0; j < disparityMap.cols; j++ )
    {
      float value = disparityMap.at<float>( i, j );
      if( value != 0 )
      {
        Vec3f point = pointcloud.at<Vec3f>( i, j );
        sum_d += plane->distance( point );
        cont++;
      }
    }
  }

  sum_d /= cont;

  // test pass if the mean of points distance from ground truth plane is lower than 3 mm
  EXPECT_LE( sum_d, 0.003 );
}

/****************************************************************************************\
*                                Test registration                                     *
 \****************************************************************************************/

TEST( GrayCodePattern, plane_reconstruction )
{
  CV_PlaneTest test;
  test.safe_run();
}
