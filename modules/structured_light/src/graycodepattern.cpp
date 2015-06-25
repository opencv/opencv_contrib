/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include <stdlib.h>
#include <iostream>

namespace cv
{
namespace structured_light
{
class CV_EXPORTS_W GrayCodePattern_Impl : public GrayCodePattern
{
public:
    // Constructor
    explicit GrayCodePattern_Impl(const GrayCodePattern::Params &parameters = GrayCodePattern::Params());

    // Destructor
    virtual ~GrayCodePattern_Impl(){};

    // Generates the gray code pattern as a std::vector<cv::Mat>
    bool generate( OutputArrayOfArrays patternImages,
                   const Scalar darkColor = Scalar(0, 0, 0),
                   const Scalar lightColor = Scalar(255, 255, 255) );

    // Decodes the gray code pattern
    bool decode( InputArrayOfArrays patternImages,
                 InputArrayOfArrays camerasMatrix,
                 InputArrayOfArrays camerasDistCoeffs,
                 InputArrayOfArrays camerasRotationMatrix,
                 InputArrayOfArrays camerasTranslationVector,
                 OutputArray disparityMap,
                 InputArrayOfArrays darkImages = noArray(),
                 InputArrayOfArrays lightImages = noArray(),
                 int flags = DECODE_3D_UNDERWORLD ) const;

    // Sets the value for black threshold
    void setDarkThreshold(int val);

    // Sets the value for set the value for white threshold
    void setLightThreshold(int val);

    void getImagesForShadowMasks(InputOutputArray darkImage, InputOutputArray lightImage) const;

private:
    // Parameters
    Params params;

    // The number of images of the pattern
    int numOfPatternImages;

    // The number of row images of the pattern
    int numOfRowImgs;

    // The number of column images of the pattern
    int numOfColImgs;

    // Number between 0-255 that represents the minimum brightness difference
    // between the fully illuminated (white) and the non - illuminated images (black)
    int darkThreshold;

    // Number between 0-255 that represents the minimum brightness difference
    // between the gray-code pattern and its inverse images
    int lightThreshold;

    Scalar _darkColor ;
    Scalar _lightColor;

    // Computes the required number of pattern images, allocating the pattern vector
    void computeNumberOfPatternImages();

    // Computes the shadows occlusion where we cannot reconstruct the model
    void computeShadowMasks(InputArrayOfArrays darkImages, InputArrayOfArrays lightImages, OutputArrayOfArrays shadowMasks) const;

    // Converts a gray code sequence to a decimal number
    int grayToDec(const std::vector<bool>& gray) const;

    // For a (x,y) pixel of the camera returns the corresponding projector pixel
    bool getProjPixel(InputArrayOfArrays patternImages, int x, int y, cv::Point &p_out) const;
};

/*
*  GrayCodePattern
*/
GrayCodePattern::Params::Params()
{
    width = 1024;
    height = 768;
}

GrayCodePattern_Impl::GrayCodePattern_Impl(const GrayCodePattern::Params &parameters) :
params(parameters)
{
  _darkColor = Scalar(0, 0, 0);
  _lightColor = Scalar(255, 255, 255);
  darkThreshold = 40;// 3D_underworld default value
  lightThreshold = 5;// 3D_underworld default value
}

bool
GrayCodePattern_Impl::generate(OutputArrayOfArrays pattern,
                               const Scalar darkColor,
                               const Scalar lightColor )
{
    _darkColor = darkColor;
   _lightColor = lightColor;

   computeNumberOfPatternImages();

    std::vector<Mat>& pattern_ = *(std::vector<Mat>*)pattern.getObj();
    pattern_.resize(numOfPatternImages);

    for (int i = 0; i < numOfPatternImages; i++)
    {
      pattern_[i] = Mat(params.height, params.width, CV_8UC3);
    }

    int flag=0;

    for (int j = 0; j < params.width; j++) // rows loop
    {
      int rem=0, num=j, prevRem=j%2;

      for (int k=0; k<numOfColImgs; k++)  // images loop
      {
        num=num/2;
        rem=num%2;

        if ((rem==0 && prevRem==1) || (rem==1 && prevRem==0))
        {
          flag = 1;
        }
        else
        {
          flag = 0;
        }

        for (int i=0;i<params.height;i++)// columns loop
        {
          Vec3b pixel_color;

          if (flag == 0)
            pixel_color = Vec3b((uchar)lightColor[0], (uchar)lightColor[1], (uchar)lightColor[2]);
          else
            pixel_color = Vec3b((uchar)(flag*darkColor[0]), (uchar)(flag*darkColor[1]), (uchar)(flag*darkColor[2]));

          pattern_[2*numOfColImgs-2*k-2].at<Vec3b>(i, j) = pixel_color;

          if (pixel_color == Vec3b((uchar)darkColor[0], (uchar)darkColor[1], (uchar)darkColor[2]))
          pixel_color = Vec3b((uchar)lightColor[0], (uchar)lightColor[1], (uchar)lightColor[2]);

          else
          pixel_color=Vec3b((uchar)darkColor[0], (uchar)darkColor[1], (uchar)darkColor[2]);

          pattern_[2*numOfColImgs-2*k-1].at<Vec3b>(i, j) = pixel_color;
        }

        prevRem=rem;
      }

    }

    for (int i = 0; i< params.height; i++)  // rows loop
    {
        int rem=0, num=i, prevRem=i%2;

        for (int k=0; k<numOfRowImgs; k++)
        {

          num=num/2;
          rem=num%2;

          if ((rem==0 && prevRem==1) || (rem==1 && prevRem==0))
          {
            flag=1;
          }
          else
          {
            flag= 0;
          }

          for (int j=0; j< params.width; j++)
          {
            cv::Vec3b pixel_color;

            if (flag==0)
              pixel_color = Vec3b((uchar)lightColor[0], (uchar)lightColor[1], (uchar)lightColor[2]);
            else
              pixel_color=Vec3b((uchar)(flag*darkColor[0]), (uchar)(flag*darkColor[1]), (uchar)(flag*darkColor[2]));

            pattern_[2*numOfRowImgs-2*k+2*numOfColImgs-2].at<Vec3b>(i, j)= pixel_color;

            if (pixel_color == Vec3b((uchar)darkColor[0], (uchar)darkColor[1], (uchar)darkColor[2]))
             pixel_color = Vec3b((uchar)lightColor[0], (uchar)lightColor[1], (uchar)lightColor[2]);

            else
             pixel_color = Vec3b((uchar)darkColor[0], (uchar)darkColor[1], (uchar)darkColor[2]);

            pattern_[2*numOfRowImgs-2*k+2*numOfColImgs-1].at<Vec3b>(i, j) = pixel_color;
          }

          prevRem=rem;
        }

      }

  return true;
}

bool
GrayCodePattern_Impl::decode( InputArrayOfArrays patternImages,
                              InputArrayOfArrays camerasMatrix,
                              InputArrayOfArrays camerasDistCoeffs,
                              InputArrayOfArrays camerasRotationMatrix,
                              InputArrayOfArrays camerasTranslationVector,
                              OutputArray disparityMap,
                              InputArrayOfArrays darkImages,
                              InputArrayOfArrays lightImages,
                              int flags ) const
{
    // To avoid unused parameters warnings
    (void) patternImages;
    (void) camerasMatrix;
    (void) camerasDistCoeffs;
    (void) camerasRotationMatrix;
    (void) camerasTranslationVector;
    (void) disparityMap;
    (void) darkImages;
    (void) lightImages;
    (void) flags;
    return true;
}

// Computes the required number of pattern images, allocating the pattern vector
void
GrayCodePattern_Impl::computeNumberOfPatternImages()
{
    numOfColImgs = (int)ceil(log(double (params.width)) / log(2.0));
    numOfRowImgs = (int)ceil(log(double (params.height)) / log(2.0));
    numOfPatternImages= 2*numOfColImgs + 2*numOfRowImgs;
}


// Computes the shadows occlusion where we cannot reconstruct the model
void
GrayCodePattern_Impl::computeShadowMasks(InputArrayOfArrays darkImages, InputArrayOfArrays lightImages, OutputArrayOfArrays shadowMasks) const
{
    (void) darkImages;
    (void) lightImages;
    (void) shadowMasks;
    return;
}

void GrayCodePattern_Impl::getImagesForShadowMasks( InputOutputArray darkImage, InputOutputArray lightImage ) const
{
      (void) darkImage;
      (void) lightImage;
}

// For a (x,y) pixel of the camera returns the corresponding projector pixel'
bool
GrayCodePattern_Impl::getProjPixel( InputArrayOfArrays patternImages, int x, int y, cv::Point &p_out ) const
{
    (void) patternImages;
    (void) x;
    (void) y;
    (void) p_out;
    return false;

}

// Converts a gray code sequence to a decimal number
int
GrayCodePattern_Impl::grayToDec( const std::vector<bool>& gray) const
{
    (void) gray;
    return 1;
}

// Sets the value for dark threshold
void
GrayCodePattern_Impl::setDarkThreshold(int val)
{
    darkThreshold = val;
}

// Sets the value for light threshold
void GrayCodePattern_Impl::setLightThreshold(int val)
{
    lightThreshold = val;
}

// Creates the GrayCodePattern instance
Ptr<GrayCodePattern> GrayCodePattern::create(const GrayCodePattern::Params& params)
{
    return makePtr<GrayCodePattern_Impl>(params);
}

}
}





