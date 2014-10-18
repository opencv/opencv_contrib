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

#ifndef __OPENCV_XFEATURES2D_FEATURES_2D_HPP__
#define __OPENCV_XFEATURES2D_FEATURES_2D_HPP__

#include "opencv2/features2d.hpp"

namespace cv
{
namespace xfeatures2d
{

/*!
 SIFT implementation.

 The class implements SIFT algorithm by D. Lowe.
*/
class CV_EXPORTS_W SIFT : public Feature2D
{
public:
    CV_WRAP static Ptr<SIFT> create( int nfeatures = 0, int nOctaveLayers = 3,
                                    double contrastThreshold = 0.04, double edgeThreshold = 10,
                                    double sigma = 1.6);
};

typedef SIFT SiftFeatureDetector;
typedef SIFT SiftDescriptorExtractor;

/*!
 SURF implementation.

 The class implements SURF algorithm by H. Bay et al.
 */
class CV_EXPORTS_W SURF : public Feature2D
{
public:
    CV_WRAP static Ptr<SURF> create(double hessianThreshold=100,
                  int nOctaves = 4, int nOctaveLayers = 3,
                  bool extended = false, bool upright = false);

    CV_WRAP virtual void setHessianThreshold(double hessianThreshold) = 0;
    CV_WRAP virtual double getHessianThreshold() const = 0;

    CV_WRAP virtual void setNOctaves(int nOctaves) = 0;
    CV_WRAP virtual int getNOctaves() const = 0;

    CV_WRAP virtual void setNOctaveLayers(int nOctaveLayers) = 0;
    CV_WRAP virtual int getNOctaveLayers() const = 0;

    CV_WRAP virtual void setExtended(bool extended) = 0;
    CV_WRAP virtual bool getExtended() const = 0;

    CV_WRAP virtual void setUpright(bool upright) = 0;
    CV_WRAP virtual bool getUpright() const = 0;
};

typedef SURF SurfFeatureDetector;
typedef SURF SurfDescriptorExtractor;

}
} /* namespace cv */

#endif
