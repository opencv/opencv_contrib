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

  // Number between 0-255 or 0-65535 that represents the minimum brightness difference
  // between the fully illuminated (white) and the non - illuminated images (black)
  size_t blackThreshold;

  // Number between 0-255 or 0-65535 that represents the minimum brightness difference
  // between the gray-code pattern and its inverse images
  size_t whiteThreshold;

  // Computes the required number of pattern images, allocating the pattern vector
  void computeNumberOfPatternImages();

  // Computes the shadows occlusion where we cannot reconstruct the model
  void computeShadowMasks( InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages,
                           OutputArrayOfArrays shadowMasks ) const;

  bool getProjPixelFast(const std::vector<Mat>& fastColImages, const std::vector<Mat>& fastRowImages, int x, int y, Point& projPix) const;
  void populateFastPatternImages(const std::vector<std::vector<Mat>>& acquiredPatterns, std::vector<std::vector<Mat>>& fastPatternColImages, std::vector<std::vector<Mat>>& fastPatternRowImages) const;
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
    std::vector<Mat>& pattern_ = *(std::vector<Mat>*)pattern.getObj();
    pattern_.resize(numOfPatternImages);
    for (size_t i = 0; i < numOfPatternImages; i++)
    {
        pattern_[i].create(params.height, params.width, CV_8U);
    }
    parallel_for_(Range(0, params.width), [&](const Range& range) {
        for (int col = range.start; col < range.end; ++col)
        {
            for (size_t nBits = 0; nBits < numOfColImgs; ++nBits)
            {
                // Gray-code column bit-plane: XOR adjacent binary bits so just one stripe
                // toggles between consecutive frames without rebuilding the Gray number.
                // It is a more efficient alternative to the decodePixel + greyToDec approach.
                // Here we get the column bit-plane for the n-th and (n+1)-th binary bits and XOR them
                // following the Gray-code definition (x_n = b_n XOR b_(n+1))
                uchar flag = (((col >> nBits) & 1) ^ ((col >> (nBits + 1)) & 1));
                uchar pixel_color = flag * 255;

                size_t idx1 = 2 * numOfColImgs - 2 * nBits - 2;
                size_t idx2 = 2 * numOfColImgs - 2 * nBits - 1;

                if (idx1 < numOfPatternImages)
                {
                    pattern_[idx1].col(col).setTo(pixel_color);
                }
                if (idx2 < numOfPatternImages)
                {
                    pattern_[idx2].col(col).setTo(255 - pixel_color);
                }
            }
        }
    });
    parallel_for_(Range(0, params.height), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            for (size_t nBits = 0; nBits < numOfRowImgs; ++nBits)
            {
                // Same Gray extraction for rows so vertical patterns flip a single stripe per exposure
                // It is a more efficient alternative to the decodePixel + greyToDec approach.
                // Here we get the row bit-plane for the n-th and (n+1)-th binary bits and XOR them
                // following the Gray-code definition (x_n = b_n XOR b_(n+1))
                uchar flag = (((row >> nBits) & 1) ^ ((row >> (nBits + 1)) & 1));
                uchar pixel_color = flag * 255;

                size_t base_idx = 2 * numOfColImgs;
                size_t idx1 = base_idx + 2 * numOfRowImgs - 2 * nBits - 2;
                size_t idx2 = base_idx + 2 * numOfRowImgs - 2 * nBits - 1;

                if (idx1 < numOfPatternImages)
                {
                    pattern_[idx1].row(row).setTo(pixel_color);
                }
                if (idx2 < numOfPatternImages)
                {
                    pattern_[idx2].row(row).setTo(255 - pixel_color);
                }
            }
        }
        });

    return true;
}


bool GrayCodePattern_Impl::decode(const std::vector< std::vector<Mat> >& patternImages, OutputArray disparityMap, InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages, int flags) const
{
    const std::vector<std::vector<Mat>>& acquired_pattern = patternImages;

    if (flags == DECODE_3D_UNDERWORLD)
    {
        // Shadow mask computation remains the same
        std::vector<Mat> shadowMasks;
        computeShadowMasks(blackImages, whiteImages, shadowMasks);

        int num_cameras = static_cast<int>(acquired_pattern.size());

        int cam_width = acquired_pattern[0][0].cols;
        int cam_height = acquired_pattern[0][0].rows;
        int proj_width = params.width;
        int proj_height = params.height;

        std::vector<std::vector<Mat>> fastPatternColImages;
        std::vector<std::vector<Mat>> fastPatternRowImages;
        populateFastPatternImages(acquired_pattern, fastPatternColImages, fastPatternRowImages);

        std::vector<Mat> projectorCoordinateMap(num_cameras);
        parallel_for_(Range(0, num_cameras), [&](const Range& range) {
            for (int k = range.start; k < range.end; k++) {
                projectorCoordinateMap[k] = Mat(cam_height, cam_width, CV_32SC2, Vec2i(-1, -1));
                Point projPixel;
                const auto& fastCol = fastPatternColImages[k];
                const auto& fastRow = fastPatternRowImages[k];
                for (int j = 0; j < cam_height; j++) {
                    for (int i = 0; i < cam_width; i++) {
                        if (shadowMasks[k].at<uchar>(j, i)) {
                            if (!getProjPixelFast(fastCol, fastRow, i, j, projPixel)) {
                                projectorCoordinateMap[k].at<Vec2i>(j, i) = Vec2i(projPixel.x, projPixel.y);
                            }
                        }
                    }
                }
            }
        });

        std::vector<Mat> sumX(num_cameras);
        std::vector<Mat> counts(num_cameras);
        for (int k = 0; k < num_cameras; ++k) {
            sumX[k] = Mat::zeros(proj_height, proj_width, CV_64F);
            counts[k] = Mat::zeros(proj_height, proj_width, CV_32S);
        }

        parallel_for_(Range(0, num_cameras), [&](const Range& range) {
            for (int k = range.start; k < range.end; ++k) {
                for (int j = 0; j < cam_height; ++j) {
                    for (int i = 0; i < cam_width; ++i) {
                        const Vec2i& projPt = projectorCoordinateMap[k].at<Vec2i>(j, i);
                        if (projPt[0] != -1) {
                            sumX[k].at<double>(projPt[1], projPt[0]) += i;
                            counts[k].at<int>(projPt[1], projPt[0]) += 1;
                        }
                    }
                }
            }
        });

        Mat counts_64F[2], avgX[2];
        counts[0].convertTo(counts_64F[0], CV_64F);
        counts[1].convertTo(counts_64F[1], CV_64F);
        cv::divide(sumX[0], counts_64F[0], avgX[0]);
        cv::divide(sumX[1], counts_64F[1], avgX[1]);

        // Handle invalid pixels (e.g. not seen by one of the cameras) by setting them to zero.
        cv::Mat invalidPixels = (counts[0] == 0) | (counts[1] == 0);

        Mat projectorDisparity = avgX[1] - avgX[0];

        Mat& disparityMap_ = *(Mat*)disparityMap.getObj();
        Mat map_x(cam_height, cam_width, CV_32F);
        Mat map_y(cam_height, cam_width, CV_32F);

        std::vector<Mat> projMapChannels;
        cv::split(projectorCoordinateMap[0], projMapChannels);
        projMapChannels[0].convertTo(map_x, CV_32F); // Projector x-coordinates
        projMapChannels[1].convertTo(map_y, CV_32F); // Projector y-coordinates

        projectorDisparity.setTo(Scalar(0.0), invalidPixels);

        cv::remap(projectorDisparity, disparityMap_, map_x, map_y,
            INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

        return true;
    }

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

  parallel_for_(Range(0, (int) whiteImages_.size()), [&](const Range& range)
  {
    for( int k = range.start; k < range.end; k++ )
    {
        cv::Mat diffImage;
        cv::absdiff(whiteImages_[k], blackImages_[k], diffImage);
        cv::compare(diffImage, static_cast<double>(blackThreshold), shadowMasks_[k], cv::CMP_GT);
        shadowMasks_[k] /= 255;
    }
  });
}

// Generates the images needed for shadowMasks computation
void GrayCodePattern_Impl::getImagesForShadowMasks( InputOutputArray blackImage, InputOutputArray whiteImage ) const
{
  Mat& blackImage_ = *( Mat* ) blackImage.getObj();
  Mat& whiteImage_ = *( Mat* ) whiteImage.getObj();

  blackImage_ = Mat( params.height, params.width, CV_8U, Scalar( 0 ) );
  whiteImage_ = Mat( params.height, params.width, CV_8U, Scalar( 255 ) );
}

void GrayCodePattern_Impl::populateFastPatternImages(const std::vector<std::vector<Mat>>& acquiredPatterns, std::vector<std::vector<Mat>>& fastPatternColImages, std::vector<std::vector<Mat>>& fastPatternRowImages) const {
    fastPatternColImages.resize(acquiredPatterns.size());
    fastPatternRowImages.resize(acquiredPatterns.size());

    for (size_t i = 0; i < acquiredPatterns.size(); i++) {
        const auto& _patternImages = acquiredPatterns[i];
        auto& fastColImagesN = fastPatternColImages[i];
        auto& fastRowImagesN = fastPatternRowImages[i];
        fastColImagesN.resize(numOfColImgs);
        fastRowImagesN.resize(numOfRowImgs);
        parallel_for_(Range(0, (int)numOfColImgs), [&](const Range& range) {
            for (int count = range.start; count < range.end; count++) {
                cv::Mat diffImage;
                cv::subtract(_patternImages[count * 2], _patternImages[count * 2 + 1], diffImage, cv::noArray(), CV_32S);
                cv::Mat invalidMask = abs(diffImage) < static_cast<double>(whiteThreshold);
                cv::compare(diffImage, cv::Scalar(0), diffImage, cv::CMP_GT);
                diffImage.setTo(1, invalidMask);
                fastColImagesN[count] = std::move(diffImage);
            }
        });

        parallel_for_(Range(0, (int)numOfRowImgs), [&](const Range& range) {
            for (int count = range.start; count < range.end; count++) {
                cv::Mat diffImage;
                cv::subtract(_patternImages[count * 2 + numOfColImgs * 2], _patternImages[count * 2 + numOfColImgs * 2 + 1], diffImage, cv::noArray(), CV_32S);
                cv::Mat invalidMask = abs(diffImage) < static_cast<double>(whiteThreshold);
                cv::compare(diffImage, cv::Scalar(0), diffImage, cv::CMP_GT);
                diffImage.setTo(1, invalidMask);
                fastRowImagesN[count] = std::move(diffImage);
            }
        });
    }
}

// NOTE: these 2 functions (grayToDec and decodePixel) are kept to keep backward compatibility with the existing API
//       since they are used in the getProjPixel function. However, the new faster method getProjPixelFast is preferred for performance
//       when decoding full patterns instead of single pixels.
static int grayToDec(const std::vector<uchar>& gray)
{
    if (gray.empty())
    {
        return 0;
    }

    int dec = 0;
    uchar prev_binary_bit = 0;
    for (size_t i = 0; i < gray.size(); ++i)
    {
        uchar current_binary_bit = prev_binary_bit ^ gray[i];
        dec = (dec << 1) | current_binary_bit;
        prev_binary_bit = current_binary_bit;
    }
    return dec;
}

template <typename T>
static bool decodePixel(int x, int y, size_t numOfColImgs, size_t numOfRowImgs, std::vector<Mat>& patternImagesVec, size_t whiteThreshold, std::vector<uchar>& grayRow, std::vector<uchar>& grayCol) {
    bool error = false;

    for (size_t k = 0; k < numOfColImgs; ++k)
    {
        T val1 = patternImagesVec[k * 2].at<T>(y, x);
        T val2 = patternImagesVec[k * 2 + 1].at<T>(y, x);

        if (std::abs(static_cast<double>(val1) - static_cast<double>(val2)) < whiteThreshold)
        {
            error = true;
        }
        grayCol[k] = (val1 > val2);
    }

    size_t base_idx = 2 * numOfColImgs;
    for (size_t k = 0; k < numOfRowImgs; ++k)
    {
        T val1 = patternImagesVec[base_idx + k * 2].at<T>(y, x);
        T val2 = patternImagesVec[base_idx + k * 2 + 1].at<T>(y, x);

        if (std::abs(static_cast<double>(val1) - static_cast<double>(val2)) < whiteThreshold)
        {
            error = true;
        }
        grayRow[k] = (val1 > val2);
    }
    return error;
}

bool GrayCodePattern_Impl::getProjPixel(InputArrayOfArrays patternImages, int x, int y, Point& projPix) const
{
    std::vector<Mat>& patternImagesVec = *(std::vector<Mat>*)patternImages.getObj();

    std::vector<uchar> grayCol(numOfColImgs);
    std::vector<uchar> grayRow(numOfRowImgs);

    bool error = false;
    int depth = patternImagesVec[0].depth();

    switch (depth)
    {
    case CV_8U:
        error = decodePixel<uchar>(x, y, numOfColImgs, numOfRowImgs, patternImagesVec, whiteThreshold, grayRow, grayCol);
        break;
    case CV_16U:
        error = decodePixel<ushort>(x, y, numOfColImgs, numOfRowImgs, patternImagesVec, whiteThreshold, grayRow, grayCol);
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Image depth not supported, only CV_8U and CV_16U");
    }

    int xDec = grayToDec(grayCol);
    int yDec = grayToDec(grayRow);

    if (yDec >= params.height || xDec >= params.width)
    {
        error = true;
    }

    projPix.x = xDec;
    projPix.y = yDec;

    return error;
}


// For a (x,y) pixel of the camera returns the corresponding projector's pixel
bool GrayCodePattern_Impl::getProjPixelFast( const std::vector<Mat>& fastColImages, const std::vector<Mat>& fastRowImages, int x, int y, Point &projPix) const
{
  int xDec = 0, yDec = 0;
  ushort binBit = 0;
  Point pattPoint = Point(x, y);

  // process column images
  for(const auto& fastColImage : fastColImages)
  {

    uchar grayBit = fastColImage.at<uchar>(pattPoint);
    // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
    if( grayBit == 1) return true;

    // determine if projection pixel is on or off
    binBit = binBit ^ (grayBit == 255);
    xDec = (xDec << 1) | binBit;
  }

  binBit = 0; // reset binary bit for row images

  // process row images
  for(const auto& fastRowImage : fastRowImages)
  {
    uchar grayBit = fastRowImage.at<uchar>(pattPoint);
    // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
    if( grayBit == 1 ) return true;

    // determine if projection pixel is on or off
    binBit = binBit ^ (grayBit == 255);
    yDec = (yDec << 1) | binBit;
  }

  if( (yDec >= params.height || xDec >= params.width) )
  {
    return true;
  }

  projPix.x = xDec;
  projPix.y = yDec;

  return false;
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
