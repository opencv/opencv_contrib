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

    // Generates the images needed for shadowMasks computation
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

    // Computes XOR operation
    bool XOR(bool val1, bool val2) const;
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
   //std::vector<Mat>& acquired_pattern = *(std::vector<Mat>*)patternImages.getObj();
   std::vector< std::vector< Mat> >& acquired_pattern = *( std::vector<std::vector<Mat> >*)patternImages.getObj();

   /* std::vector< std::vector <cv::Mat > > pattern;
   pattern.resize(2);

   for (int i=0; i<numOfPatternImages;i++)
   {
        pattern[0][i]=acquired_pattern[i];
        pattern[1][i]=acquired_pattern[i+numOfPatternImages];
    }*/

    // Computing shadows mask
    std::vector<Mat> shadowMasks;
    computeShadowMasks(darkImages, lightImages, shadowMasks);

    //int cam_width = acquired_pattern[0].cols;
    //int cam_height = acquired_pattern[0].rows;

    int cam_width = acquired_pattern[0][0].cols;
    int cam_height = acquired_pattern[0][0].rows;

    cv::Point projPixel;

    std::vector<cv::Point> **camsPixels = new std::vector<cv::Point>*[acquired_pattern.size()];
    std::vector<cv::Point>* camPixels;// =new std::vector<cv::Point>[params.height*params.width];
    std::vector<Mat> decoded;
    decoded.resize(4);

    for(int k=0; k < (int) acquired_pattern.size(); k++)
    {
         camsPixels[k]= new std::vector<cv::Point>[params.height*params.width];
         camPixels = camsPixels[k];

         decoded[2*k] = Mat(cam_height, cam_width, CV_64F);
         decoded[2*k+1] = Mat(cam_height, cam_width, CV_64F);
         for(int i=0; i<cam_width; i++)
         {
            for(int j=0; j<cam_height; j++)
            {
              //if the pixel is not shadow reconstruct
              if(shadowMasks[k].at<uchar>(j,i))
              {
                  //get the projector pixel for camera (i,j) pixel
                  //for a (x,y) pixel of the camera returns the corresponding projector pixel by calculating the decimal number
                  //projpixel contains the decimal number corresponding to the row and column gray code
                  bool error = getProjPixel(acquired_pattern[k], i,j,projPixel);// il problema è qui: devo passare metà pattern e metà èat
                  //bool error = getProjPixel(pattern[k], i,j,projPixel);

                  if(error)
                  {
                    shadowMasks[k].at<uchar>(j,i) = 0;
                    continue;
                  }

                  camPixels[ projPixel.x*params.height + projPixel.y].push_back(cv::Point(i,j));
                  decoded[2*k].at<double>(j,i)=(double)projPixel.x;
                  decoded[2*k+1].at<double>(j,i)=(double)projPixel.y;
              }
            }
         }
     }

     std::vector<cv::Point> cam1Pixs,cam2Pixs;

     Mat& disparityMap_ = *( Mat*)disparityMap.getObj();
     disparityMap_ = Mat(params.height, params.width, CV_64F);


     for(int i=0; i< params.width; i++)
     {
         for(int j=0; j< params.height; j++)
         {
            cam1Pixs = camsPixels[0][i*params.height + j];
            cam2Pixs = camsPixels[1][i*params.height + j];

            if( cam1Pixs.size() == 0 || cam2Pixs.size() == 0)
                    continue;
            for(int c1=0; c1 < (int)cam1Pixs.size(); c1++)
            {
              cv::Point p1 = cam1Pixs[c1];
              for(int c2=0; c2 < (int)cam2Pixs.size(); c2++)
              {
                cv::Point p2 = cam2Pixs[c2];
                double disp = std::sqrt((double)((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)));
                disparityMap_.at<double>(j,i)=(double)disp;
              }

            }


         }

       }




     // Debug purposes: to visualize the results
     cv::Size ns = cv::Size(640, 480);

     resize(decoded[0], decoded[0], ns);
     resize(decoded[1], decoded[1], ns);
     resize(decoded[2], decoded[2], ns);
     resize(decoded[3], decoded[3], ns);

     /*imshow("cam 1 x", decoded[0]);
       imshow("cam 1 y", decoded[1]);
       imshow("cam 2 x", decoded[2]);
       imshow("cam 2 y", decoded[3]);*/

       // Apply the colormap:
       double min;
       double max;

       cv::minMaxIdx(decoded[0], &min, &max);
       cv::convertScaleAbs(decoded[0], decoded[0], 255 / max);
       cv::minMaxIdx(decoded[1], &min, &max);
       cv::convertScaleAbs(decoded[1], decoded[1], 255 / max);
       cv::minMaxIdx(decoded[2], &min, &max);
       cv::convertScaleAbs(decoded[2], decoded[2], 255 / max);
       cv::minMaxIdx(decoded[3], &min, &max);
       cv::convertScaleAbs(decoded[3], decoded[3], 255 / max);
       cv::Mat cm_img0, cm_img1, cm_img2, cm_img3;
       applyColorMap(decoded[0], cm_img0, COLORMAP_JET);
       applyColorMap(decoded[1], cm_img1, COLORMAP_JET);
       applyColorMap(decoded[2], cm_img2, COLORMAP_JET);
       applyColorMap(decoded[3], cm_img3, COLORMAP_JET);
       // Show the result:
       imshow("cm cam 1 x", cm_img0);
       imshow("cm cam 1 y", cm_img1);
       imshow("cm cam 2 x", cm_img2);
       imshow("cm cam 2 y", cm_img3);




    // To avoid unused parameters warnings
    (void) camPixels;
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
    std::vector<Mat>& lightImages_ = *(std::vector<Mat>*) lightImages.getObj();
    std::vector<Mat>& darkImages_ = *(std::vector<Mat>*) darkImages.getObj();
    std::vector<Mat>& shadowMasks_ = *(std::vector<Mat>*) shadowMasks.getObj();

    shadowMasks_.resize(lightImages_.size());

    int cam_width = lightImages_[0].cols;
    int cam_height = lightImages_[0].rows;

    for (int k = 0; k < (int) shadowMasks_.size(); k++)
    {
          shadowMasks_[k] = Mat(cam_height, cam_width, CV_8U);
          for(int i=0; i < cam_width; i++)
          {
                for(int j=0; j < cam_height; j++)
                {
                  // if we read the images as color
                  //Vec3b lightColor = lightImages_[k].at<Vec3b>(Point(i,j));
                  //Vec3b darkColor = darkImages_[k].at<Vec3b>(Point(i,j));

                  uchar lightColor = lightImages_[k].at<uchar>(Point(i,j));
                  uchar darkColor = darkImages_[k].at<uchar>(Point(i,j));

                  //Vec3b ciao = lightColor - darkColor;//if(ciao[0] > darkThreshold)//
                  if(lightColor - darkColor > darkThreshold)// this works if I read the images as grayscale
                  {
                     shadowMasks_[k].at<uchar>(Point(i,j)) =(uchar)1;
                  }
                  else
                  {
                    shadowMasks_[k].at<uchar>(Point(i,j)) =(uchar)0;
                  }
                }
           }

       // Only for debug purpose
       cv::Size ns = cv::Size(640, 480);
       std::ostringstream  name;
       name << "shadows cam" << k ;
       cv::Mat tmp;
       resize(shadowMasks_[k], tmp, ns);
       imshow(name.str(), 255*tmp);

     }

}

// Generates the images needed for shadowMasks computation
void
GrayCodePattern_Impl::getImagesForShadowMasks( InputOutputArray darkImage, InputOutputArray lightImage ) const
{
    Mat& darkImage_ = *(Mat*)darkImage.getObj();
    Mat& lightImage_ = *(Mat*)lightImage.getObj();

    darkImage_ = Mat(params.height, params.width, CV_8UC3, _darkColor);
    lightImage_= Mat(params.height, params.width, CV_8UC3, _lightColor);
}

// For a (x,y) pixel of the camera returns the corresponding projector pixel'
bool
GrayCodePattern_Impl::getProjPixel( InputArrayOfArrays patternImages, int x, int y, cv::Point &p_out ) const
{
    std::vector<Mat>& _patternImages = *(std::vector<Mat>*)patternImages.getObj();
    std::vector<bool> grayCol;
    std::vector<bool> grayRow;

    bool error = false;
    int xDec, yDec;

    //process column images
    for(int count=0; count<numOfRowImgs; count++)
    {
     //get pixel intensity for regular pattern projection and its inverse
     double val1, val2;
     val1 = _patternImages[count * 2].at<uchar>(Point(x,y));//   Utilities::matGet2D(camImgs[count * 2 + 2   ],x,y);
     val2 = _patternImages[count * 2 +1 ].at<uchar>(Point(x,y));//Utilities::matGet2D(camImgs[count * 2 + 2 +1],x,y);

     //check if intensity deference is in a valid rage
     if(abs(val1-val2) < lightThreshold )
       error=true;

     //determine if projection pixel is on or off
     if(val1>val2)
       grayCol.push_back(1);

     else
       grayCol.push_back(0);

    }

    xDec = grayToDec(grayCol);

    //process row images
    for(int count=0; count < numOfColImgs; count++)
    {
      double val1, val2;

      val1 = _patternImages[count * 2 +numOfColImgs*2 ].at<uchar>(Point(x,y));//Utilities::matGet2D(camImgs[count*2+2+numOfColBits*2],x,y);
      val2 = _patternImages[count * 2 +numOfColImgs*2 +1].at<uchar>(Point(x,y));; //Utilities::matGet2D(camImgs[count*2+2+numOfColBits*2+1],x,y);

      if(abs(val1-val2) < lightThreshold )  //check if the difference between the values of the normal and it's inverse projection image is valid
       error = true;

      if(val1 > val2)
       grayRow.push_back(1);
      else
       grayRow.push_back(0);

    }

    //decode
    yDec = grayToDec(grayRow);

    if((yDec > params.height || xDec > params.width))
    {
      error = true;
    }

    p_out.x = xDec;
    p_out.y = yDec;

    return error;
}

// Converts a gray code sequence (binary number) to a decimal number
int
GrayCodePattern_Impl::grayToDec( const std::vector<bool>& gray) const
{
    (void) gray;
    int dec=0;
    bool tmp = gray[0];

    if(tmp)
      dec+=(int) pow((float)2,int (gray.size() -1));

    for(int i = 1; i < (int) gray.size(); i++)
    {
        tmp=XOR(tmp,gray[i]);

        if(tmp)
          dec+= (int) pow((float)2,int (gray.size()-i-1) );
    }

    return dec;
}

// Computes XOR operation
bool
GrayCodePattern_Impl::XOR(bool val1, bool val2) const
{
  if(val1==val2)
    return 0;
  else
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





