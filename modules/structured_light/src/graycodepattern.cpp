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

#include "precomp.hpp"

namespace cv {
namespace structured_light {
class CV_EXPORTS_W GrayCodePattern_Impl CV_FINAL : public GrayCodePattern
{
 public:
  // Constructor
  explicit GrayCodePattern_Impl( const GrayCodePattern::Params &parameters = GrayCodePattern::Params() );

  // Destructor
  virtual ~GrayCodePattern_Impl() CV_OVERRIDE {};

  // Generates the gray code pattern as a std::vector<Mat>
  bool generate( OutputArrayOfArrays patternImages ) CV_OVERRIDE;

  // Decodes the gray code pattern, computing the disparity map
  bool decode( const std::vector< std::vector<Mat> >& patternImages, OutputArray disparityMap, InputArrayOfArrays blackImages = noArray(),
               InputArrayOfArrays whiteImages = noArray(), int flags = DECODE_3D_UNDERWORLD ) const CV_OVERRIDE;

  // Returns the number of pattern images for the graycode pattern
  size_t getNumberOfPatternImages() const CV_OVERRIDE;

  // Sets the value for black threshold
  void setBlackThreshold( size_t val ) CV_OVERRIDE;

  // Sets the value for set the value for white threshold
  void setWhiteThreshold( size_t val ) CV_OVERRIDE;

  // Generates the images needed for shadowMasks computation
  void getImagesForShadowMasks( InputOutputArray blackImage, InputOutputArray whiteImage ) const CV_OVERRIDE;

  // For a (x,y) pixel of the camera returns the corresponding projector pixel
  bool getProjPixel(InputArrayOfArrays patternImages, int x, int y, CV_OUT Point &projPix) const CV_OVERRIDE;

 private:
  // Parameters
  Params params;

  // The number of images of the pattern
  size_t numOfPatternImages;

  // The number of row images of the pattern
  size_t numOfRowImgs;

  // The number of column images of the pattern
  size_t numOfColImgs;

  // Number between 0-255 that represents the minimum brightness difference
  // between the fully illuminated (white) and the non - illuminated images (black)
  size_t blackThreshold;

  // Number between 0-255 that represents the minimum brightness difference
  // between the gray-code pattern and its inverse images
  size_t whiteThreshold;

  // Computes the required number of pattern images, allocating the pattern vector
  void computeNumberOfPatternImages();

  // Computes the shadows occlusion where we cannot reconstruct the model
  void computeShadowMasks( InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages,
                           OutputArrayOfArrays shadowMasks ) const;

  // Converts a gray code sequence (~ binary number) to a decimal number
  int grayToDec( const std::vector<uchar>& gray ) const;
};

/*
 *  GrayCodePattern
 */
GrayCodePattern::Params::Params()
{
  width = 1024;
  height = 768;
}

GrayCodePattern_Impl::GrayCodePattern_Impl( const GrayCodePattern::Params &parameters ) :
    params( parameters )
{
  computeNumberOfPatternImages();
  blackThreshold = 40;  // 3D_underworld default value
  whiteThreshold = 5;   // 3D_underworld default value
}

bool GrayCodePattern_Impl::generate( OutputArrayOfArrays pattern )
{
  std::vector<Mat>& pattern_ = *( std::vector<Mat>* ) pattern.getObj();
  pattern_.resize( numOfPatternImages );

  for( size_t i = 0; i < numOfPatternImages; i++ )
  {
    pattern_[i] = Mat( params.height, params.width, CV_8U );
  }

  uchar flag = 0;

  for( int j = 0; j < params.width; j++ )  // rows loop
  {
    int rem = 0, num = j, prevRem = j % 2;

    for( size_t k = 0; k < numOfColImgs; k++ )  // images loop
    {
      num = num / 2;
      rem = num % 2;

      if( ( rem == 0 && prevRem == 1 ) || ( rem == 1 && prevRem == 0) )
      {
        flag = 1;
      }
      else
      {
        flag = 0;
      }

      for( int i = 0; i < params.height; i++ )  // rows loop
      {

        uchar pixel_color = ( uchar ) flag * 255;

        pattern_[2 * numOfColImgs - 2 * k - 2].at<uchar>( i, j ) = pixel_color;
        if( pixel_color > 0 )
          pixel_color = ( uchar ) 0;
        else
         pixel_color = ( uchar ) 255;
        pattern_[2 * numOfColImgs - 2 * k - 1].at<uchar>( i, j ) = pixel_color;  // inverse
      }

      prevRem = rem;
    }
  }

  for( int i = 0; i < params.height; i++ )  // rows loop
  {
    int rem = 0, num = i, prevRem = i % 2;

    for( size_t k = 0; k < numOfRowImgs; k++ )
    {
      num = num / 2;
      rem = num % 2;

      if( (rem == 0 && prevRem == 1) || (rem == 1 && prevRem == 0) )
      {
        flag = 1;
      }
      else
      {
        flag = 0;
      }

      for( int j = 0; j < params.width; j++ )
      {

        uchar pixel_color = ( uchar ) flag * 255;
        pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 2].at<uchar>( i, j ) = pixel_color;

        if( pixel_color > 0 )
          pixel_color = ( uchar ) 0;
        else
          pixel_color = ( uchar ) 255;

        pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 1].at<uchar>( i, j ) = pixel_color;
      }

      prevRem = rem;
    }
  }

  return true;
}

bool GrayCodePattern_Impl::decode( const std::vector< std::vector<Mat> >& patternImages, OutputArray disparityMap,
                                   InputArrayOfArrays blackImages, InputArrayOfArrays whitheImages, int flags ) const
{
  const std::vector<std::vector<Mat> >& acquired_pattern = patternImages;

  if( flags == DECODE_3D_UNDERWORLD )
  {
    // Computing shadows mask
    std::vector<Mat> shadowMasks;
    computeShadowMasks( blackImages, whitheImages, shadowMasks );

    int cam_width = acquired_pattern[0][0].cols;
    int cam_height = acquired_pattern[0][0].rows;

    Point projPixel;

    // Storage for the pixels of the two cams that correspond to the same pixel of the projector
    std::vector<std::vector<std::vector<Point> > > camsPixels;
    camsPixels.resize( acquired_pattern.size() );

    // TODO: parallelize for (k and j)
    for( size_t k = 0; k < acquired_pattern.size(); k++ )
    {
      camsPixels[k].resize( params.height * params.width );
      for( int i = 0; i < cam_width; i++ )
      {
        for( int j = 0; j < cam_height; j++ )
        {
          //if the pixel is not shadowed, reconstruct
          if( shadowMasks[k].at<uchar>( j, i ) )
          {
            //for a (x,y) pixel of the camera returns the corresponding projector pixel by calculating the decimal number
            bool error = getProjPixel( acquired_pattern[k], i, j, projPixel );

            if( error )
            {
              continue;
            }

            camsPixels[k][projPixel.x * params.height + projPixel.y].push_back( Point( i, j ) );
          }
        }
      }
    }

    std::vector<Point> cam1Pixs, cam2Pixs;

    Mat& disparityMap_ = *( Mat* ) disparityMap.getObj();
    disparityMap_ = Mat( cam_height, cam_width, CV_64F, double( 0 ) );

    double number_of_pixels_cam1 = 0;
    double number_of_pixels_cam2 = 0;

    for( int i = 0; i < params.width; i++ )
    {
      for( int j = 0; j < params.height; j++ )
      {
        cam1Pixs = camsPixels[0][i * params.height + j];
        cam2Pixs = camsPixels[1][i * params.height + j];

        if( cam1Pixs.size() == 0 || cam2Pixs.size() == 0 )
          continue;

        Point p1;
        Point p2;

        double sump1x = 0;
        double sump2x = 0;

        number_of_pixels_cam1 += cam1Pixs.size();
        number_of_pixels_cam2 += cam2Pixs.size();
        for( int c1 = 0; c1 < (int) cam1Pixs.size(); c1++ )
        {
          p1 = cam1Pixs[c1];
          sump1x += p1.x;
        }
        for( int c2 = 0; c2 < (int) cam2Pixs.size(); c2++ )
        {
          p2 = cam2Pixs[c2];
          sump2x += p2.x;
        }

        sump2x /= cam2Pixs.size();
        sump1x /= cam1Pixs.size();
        for( int c1 = 0; c1 < (int) cam1Pixs.size(); c1++ )
        {
          p1 = cam1Pixs[c1];
          disparityMap_.at<double>( p1.y, p1.x ) = ( double ) (sump2x - sump1x);
        }

        sump2x = 0;
        sump1x = 0;
      }
    }

    return true;
  }  // end if flags

  return false;
}

// Computes the required number of pattern images
void GrayCodePattern_Impl::computeNumberOfPatternImages()
{
  numOfColImgs = ( size_t ) ceil( log( double( params.width ) ) / log( 2.0 ) );
  numOfRowImgs = ( size_t ) ceil( log( double( params.height ) ) / log( 2.0 ) );
  numOfPatternImages = 2 * numOfColImgs + 2 * numOfRowImgs;
}

// Returns the number of pattern images to project / decode
size_t GrayCodePattern_Impl::getNumberOfPatternImages() const
{
  return numOfPatternImages;
}

// Computes the shadows occlusion where we cannot reconstruct the model
void GrayCodePattern_Impl::computeShadowMasks( InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages,
                                               OutputArrayOfArrays shadowMasks ) const
{
  std::vector<Mat>& whiteImages_ = *( std::vector<Mat>* ) whiteImages.getObj();
  std::vector<Mat>& blackImages_ = *( std::vector<Mat>* ) blackImages.getObj();
  std::vector<Mat>& shadowMasks_ = *( std::vector<Mat>* ) shadowMasks.getObj();

  shadowMasks_.resize( whiteImages_.size() );

  int cam_width = whiteImages_[0].cols;
  int cam_height = whiteImages_[0].rows;

  // TODO: parallelize for
  for( int k = 0; k < (int) shadowMasks_.size(); k++ )
  {
    shadowMasks_[k] = Mat( cam_height, cam_width, CV_8U );
    for( int i = 0; i < cam_width; i++ )
    {
      for( int j = 0; j < cam_height; j++ )
      {
        double white = whiteImages_[k].at<uchar>( Point( i, j ) );
        double black = blackImages_[k].at<uchar>( Point( i, j ) );

        if( abs(white - black) > blackThreshold )
        {
          shadowMasks_[k].at<uchar>( Point( i, j ) ) = ( uchar ) 1;
        }
        else
        {
          shadowMasks_[k].at<uchar>( Point( i, j ) ) = ( uchar ) 0;
        }
      }
    }
  }
}

// Generates the images needed for shadowMasks computation
void GrayCodePattern_Impl::getImagesForShadowMasks( InputOutputArray blackImage, InputOutputArray whiteImage ) const
{
  Mat& blackImage_ = *( Mat* ) blackImage.getObj();
  Mat& whiteImage_ = *( Mat* ) whiteImage.getObj();

  blackImage_ = Mat( params.height, params.width, CV_8U, Scalar( 0 ) );
  whiteImage_ = Mat( params.height, params.width, CV_8U, Scalar( 255 ) );
}

// For a (x,y) pixel of the camera returns the corresponding projector's pixel
bool GrayCodePattern_Impl::getProjPixel( InputArrayOfArrays patternImages, int x, int y, Point &projPix ) const
{
  std::vector<Mat>& _patternImages = *( std::vector<Mat>* ) patternImages.getObj();
  std::vector<uchar> grayCol;
  std::vector<uchar> grayRow;

  bool error = false;
  int xDec, yDec;

  // process column images
  for( size_t count = 0; count < numOfColImgs; count++ )
  {
    // get pixel intensity for regular pattern projection and its inverse
    double val1 = _patternImages[count * 2].at<uchar>( Point( x, y ) );
    double val2 = _patternImages[count * 2 + 1].at<uchar>( Point( x, y ) );

    // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
    if( abs(val1 - val2) < whiteThreshold )
      error = true;

    // determine if projection pixel is on or off
    if( val1 > val2 )
      grayCol.push_back( 1 );
    else
      grayCol.push_back( 0 );
  }

  xDec = grayToDec( grayCol );

  // process row images
  for( size_t count = 0; count < numOfRowImgs; count++ )
  {
    // get pixel intensity for regular pattern projection and its inverse
    double val1 = _patternImages[count * 2 + numOfColImgs * 2].at<uchar>( Point( x, y ) );
    double val2 = _patternImages[count * 2 + numOfColImgs * 2 + 1].at<uchar>( Point( x, y ) );

    // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
    if( abs(val1 - val2) < whiteThreshold )
      error = true;

    // determine if projection pixel is on or off
    if( val1 > val2 )
      grayRow.push_back( 1 );
    else
      grayRow.push_back( 0 );
  }

  yDec = grayToDec( grayRow );

  if( (yDec >= params.height || xDec >= params.width) )
  {
    error = true;
  }

  projPix.x = xDec;
  projPix.y = yDec;

  return error;
}

// Converts a gray code sequence (~ binary number) to a decimal number
int GrayCodePattern_Impl::grayToDec( const std::vector<uchar>& gray ) const
{
  int dec = 0;

  uchar tmp = gray[0];

  if( tmp )
    dec += ( int ) pow( ( float ) 2, int( gray.size() - 1 ) );

  for( int i = 1; i < (int) gray.size(); i++ )
  {
    // XOR operation
    tmp = tmp ^ gray[i];
    if( tmp )
      dec += (int) pow( ( float ) 2, int( gray.size() - i - 1 ) );
  }

  return dec;
}

// Sets the value for black threshold
void GrayCodePattern_Impl::setBlackThreshold( size_t val )
{
  blackThreshold = val;
}

// Sets the value for white threshold
void GrayCodePattern_Impl::setWhiteThreshold( size_t val )
{
  whiteThreshold = val;
}

// Creates the GrayCodePattern instance
Ptr<GrayCodePattern> GrayCodePattern::create( const GrayCodePattern::Params& params )
{
  return makePtr<GrayCodePattern_Impl>( params );
}

// Creates the GrayCodePattern instance
// alias for scripting
Ptr<GrayCodePattern> GrayCodePattern::create( int width, int height )
{
  Params params;
  params.width = width;
  params.height = height;
  return makePtr<GrayCodePattern_Impl>( params );
}

}
}
