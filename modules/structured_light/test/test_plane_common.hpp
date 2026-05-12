// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_TEST_PLANE_COMMON_HPP__
#define __OPENCV_TEST_PLANE_COMMON_HPP__

#include "test_precomp.hpp"

#include <cmath>
#include <limits>

namespace opencv_test { namespace {

const string STRUCTURED_LIGHT_DIR = "structured_light";
const string FOLDER_DATA = "data";

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

  virtual float distance(const Vec3f& p_j) const = 0;

  inline float d() const
  {
    return d_;
  }

  const Vec3f &
  n() const
  {
    return n_;
  }

  void UpdateParameters()
  {
    if( empty() )
      return;
    m_ = m_sum_ / K_;
    Matx33f C = Q_ - m_sum_ * m_.t();
    SVD svd(C);
    n_ = Vec3f(svd.vt.at<float>(2, 0), svd.vt.at<float>(2, 1), svd.vt.at<float>(2, 2));
    mse_ = svd.w.at<float>(2) / K_;
    UpdateD();
  }

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
  int index_;
 protected:
  float d_;
  Vec3f n_;
 private:
  inline void UpdateD()
  {
    d_ = -m_.dot(n_);
  }
  Vec3f m_sum_;
  Vec3f m_;
  Matx33f Q_;
  Matx33f C_;
  float mse_;
  int K_;
};

class Plane : public PlaneBase
{
 public:
  Plane(const Vec3f & m, const Vec3f &n_in, int index) :
      PlaneBase(m, n_in, index)
  {
  }

  float distance(const Vec3f& p_j) const CV_OVERRIDE
  {
    return std::abs(float(p_j.dot(n_) + d_));
  }
};

template<bool Use16Bit>
class CV_PlaneTest : public cvtest::BaseTest
{
 public:
  CV_PlaneTest() = default;
  ~CV_PlaneTest() CV_OVERRIDE = default;

 protected:
  void run( int ) CV_OVERRIDE
  {
    string folder = cvtest::TS::ptr()->get_data_path() + "/" + STRUCTURED_LIGHT_DIR + "/" + FOLDER_DATA + "/";
    structured_light::GrayCodePattern::Params params;
    params.width = 1280;
    params.height = 800;
    Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create( params );
    size_t numberOfPatternImages = graycode->getNumberOfPatternImages();

    FileStorage fs( folder + "calibrationParameters.yml", FileStorage::READ );
    if( !fs.isOpened() )
    {
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    FileStorage fs2( folder + "gt_plane.yml", FileStorage::READ );
    if( !fs2.isOpened() )
    {
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    Vec4f plane_coefficients;
    Vec3f m;
    fs2["plane_coefficients"] >> plane_coefficients;
    fs2["m"] >> m;

    Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;

    fs["cam1_intrinsics"] >> cam1intrinsics;
    fs["cam2_intrinsics"] >> cam2intrinsics;
    fs["cam1_distorsion"] >> cam1distCoeffs;
    fs["cam2_distorsion"] >> cam2distCoeffs;
    fs["R"] >> R;
    fs["T"] >> T;

    vector<Mat> blackImages(2);
    vector<Mat> whiteImages(2);

    whiteImages[0] = imread( folder + "pattern_cam1_im43.jpg", IMREAD_GRAYSCALE );
    whiteImages[1] = imread( folder + "pattern_cam2_im43.jpg", IMREAD_GRAYSCALE );
    blackImages[0] = imread( folder + "pattern_cam1_im44.jpg", IMREAD_GRAYSCALE );
    blackImages[1] = imread( folder + "pattern_cam2_im44.jpg", IMREAD_GRAYSCALE );

    if( Use16Bit )
    {
      for( int i = 0; i < 2; i++ )
      {
        whiteImages[i].convertTo( whiteImages[i], CV_16U, 257.0 );
        blackImages[i].convertTo( blackImages[i], CV_16U, 257.0 );
      }
    }

    Size imagesSize = whiteImages[0].size();

    if( ( !cam1intrinsics.data ) || ( !cam2intrinsics.data ) || ( !cam1distCoeffs.data ) || ( !cam2distCoeffs.data ) || ( !R.data )
        || ( !T.data ) || ( !whiteImages[0].data ) || ( !whiteImages[1].data ) || ( !blackImages[0].data )
        || ( !blackImages[1].data ) )
    {
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    stereoRectify( cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, 0,
                   -1, imagesSize, &validRoi[0], &validRoi[1] );

    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap( cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y );
    initUndistortRectifyMap( cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y );

    vector<vector<Mat> > captured_pattern( 2, std::vector<Mat>( numberOfPatternImages ) );

    for( size_t i = 0; i < numberOfPatternImages; i++ )
    {
      std::ostringstream name1;
      name1 << "pattern_cam1_im" << i + 1 << ".jpg";
      captured_pattern[0][i] = imread( folder + name1.str(), IMREAD_GRAYSCALE );
      std::ostringstream name2;
      name2 << "pattern_cam2_im" << i + 1 << ".jpg";
      captured_pattern[1][i] = imread( folder + name2.str(), IMREAD_GRAYSCALE );

      if( Use16Bit )
      {
        captured_pattern[0][i].convertTo( captured_pattern[0][i], CV_16U, 257.0 );
        captured_pattern[1][i].convertTo( captured_pattern[1][i], CV_16U, 257.0 );
      }

      if( (!captured_pattern[0][i].data) || (!captured_pattern[1][i].data) )
      {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
      }

      remap( captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
      remap( captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    }

    remap( whiteImages[0], whiteImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    remap( whiteImages[1], whiteImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );

    remap( blackImages[0], blackImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    remap( blackImages[1], blackImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );

    const double scale = Use16Bit ? 257.0 : 1.0;
    const size_t blackThreshold = static_cast<size_t>(55 * scale);
    const size_t whiteThreshold = static_cast<size_t>(10 * scale);
    graycode->setBlackThreshold( blackThreshold );
    graycode->setWhiteThreshold( whiteThreshold );

    for( int cam = 0; cam < 2; ++cam )
    {
      Mat diffImage, mask;
      absdiff( whiteImages[cam], blackImages[cam], diffImage );
      cv::compare( diffImage, Scalar( static_cast<double>(blackThreshold) ), mask, CMP_GT );
      EXPECT_GT( countNonZero( mask ), 0 ) << "Shadow mask empty for camera " << cam;
    }

    Mat disparityMap;
    bool decoded = graycode->decode( captured_pattern, disparityMap, blackImages, whiteImages,
                                     structured_light::DECODE_3D_UNDERWORLD );
    ASSERT_TRUE( decoded );
    ASSERT_FALSE( disparityMap.empty() );

    double minVal = 0, maxVal = 0;
    minMaxLoc( disparityMap, &minVal, &maxVal );
    EXPECT_LT( minVal, maxVal );

    Mat pointcloud;
    disparityMap.convertTo( disparityMap, CV_32FC1 );
    reprojectImageTo3D( disparityMap, pointcloud, Q, true, -1 );
    pointcloud = pointcloud / 1000;

    Vec3f normal( plane_coefficients.val[0], plane_coefficients.val[1], plane_coefficients.val[2] );
    Ptr<PlaneBase> plane = Ptr<PlaneBase>( new Plane( m, normal, 0 ) );

    float sum_d = 0;
    int cont = 0;
    for( int i = 0; i < disparityMap.rows; i++ )
    {
      for( int j = 0; j < disparityMap.cols; j++ )
      {
        float disparityValue = disparityMap.at<float>( i, j );
        if( disparityValue <= std::numeric_limits<float>::epsilon() )
          continue;

        Vec3f point = pointcloud.at<Vec3f>( i, j );

        if( std::isfinite( point[0] ) && std::isfinite( point[1] ) && std::isfinite( point[2] ) &&
            std::abs( point[2] ) <= 10.0f )
        {
          sum_d += plane->distance( point );
          cont++;
        }
      }
    }

    ASSERT_GT( cont, 0 );
    float mean_distance = sum_d / cont;

    EXPECT_LE( mean_distance, 0.003f );
  }
};

}} // namespace

#endif // __OPENCV_TEST_PLANE_COMMON_HPP__