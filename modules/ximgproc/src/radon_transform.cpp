/****************************************************************************************
 * By downloading, copying, installing or using the software you agree to this license. *
 * If you do not agree to this license, do not download, install,                       *
 * copy or use the software.                                                            *
 *                                                                                      *
 *                                                                                      *
 *                         License Agreement                                            *
 *              For Open Source Computer Vision Library                                 *
 *                     (3-clause BSD License)                                           *
 *                                                                                      *
 * Copyright (C) 2000-2016, Intel Corporation, all rights reserved.                     *
 * Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.                    *
 * Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.                    *
 * Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.          *
 * Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.                     *
 * Copyright (C) 2015-2016, Itseez Inc., all rights reserved.                           *
 * Third party copyrights are property of their respective owners.                      *
 *                                                                                      *
 * Redistribution and use in source and binary forms, with or without modification,     *
 * are permitted provided that the following conditions are met:                        *
 *                                                                                      *
 * Redistributions of source code must retain the above copyright notice,               *
 * this list of conditions and the following disclaimer.                                *
 *                                                                                      *
 * Redistributions in binary form must reproduce the above copyright notice,            *
 * this list of conditions and the following disclaimer in the documentation            *
 * and/or other materials provided with the distribution.                               *
 *                                                                                      *
 * Neither the names of the copyright holders nor the names of the contributors         *
 * may be used to endorse or promote products derived from this software                *
 * without specific prior written permission.                                           *
 *                                                                                      *
 * This software is provided by the copyright holders and contributors "as is" and      *
 * any express or implied warranties, including, but not limited to, the implied        *
 * warranties of merchantability and fitness for a particular purpose are disclaimed.   *
 * In no event shall copyright holders or contributors be liable for any direct,        *
 * indirect, incidental, special, exemplary, or consequential damages                   *
 * (including, but not limited to, procurement of substitute goods or services;         *
 * loss of use, data, or profits; or business interruption) however caused              *
 * and on any theory of liability, whether in contract, strict liability,               *
 * or tort (including negligence or otherwise) arising in any way out of                *
 * the use of this software, even if advised of the possibility of such damage.         *
 ****************************************************************************************/

 /**
  * @author {aravind | arvindsuresh2009@gmail.com}
  * Created on Sat, Feb 13 2016
  */

 /**
  * Implementation of "The Radon Transform - Theory and Implementation"
  *   						 by "Toft, {Peter Aundal} and Sørensen, {John Aasted}"
  *
  * Online publication: http://petertoft.dk/PhD/
  */

// Necessary headers

#include "precomp.hpp"

#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include "opencv2/ximgproc.hpp"

#ifdef __cplusplus
namespace cv {
  namespace ximgproc {
#endif

/**
 * [trans Function for computing radon transform given a particular angle theta]
 * @param src        [Source image]
 * @param dst        [Destination image]
 * @param m          [Half the columns in src]
 * @param n          [Half the rows in src]
 * @param rhoMax     [Maximum value of rho ( restricted due to finite dimension of the image )]
 * @param theta      [Given value of theta]
 * @param minAngle   [Minimum values of angle for which transform is computed]
 */

template <typename T, int D>
static inline void trans( Mat src,
                          Mat & dst,
                          int m,
                          int n,
                          int rhoMax,
                          int theta,
                          int minAngle
                        ) {
  int rc = rhoMax/2;

  // Computing cos(theta) and sin(theta)
  float cosTh = static_cast<float>(std::cos(theta*CV_PI/180.0));
  float sinTh = static_cast<float>(std::sin(theta*CV_PI/180.0));

  // Slope of the line = tan(90 - theta)
  float a = -cosTh/sinTh;
    // Range: [1, 45]
    if( theta <= 45 ) {
      for(int r = 1; r <= rhoMax; ++r) {
        float rho = (float)(r - rc);
        float b = rho/sinTh;

        int yMax = (std::min((int)(-a*m + b), n-1)),
            yMin = (std::max((int)(a*m + b), -n));

        for(int y = yMin; y <= yMax; ++y) {
          float x = (y - b)/a;
          int xFloor = static_cast<int>(std::floor(x));
          float xUp = x - xFloor;
          float xLow = 1.0f - xUp;
          x = (float)(xFloor);
          x = static_cast<float>(std::max((int)(x), -m));
          x = static_cast<float>(std::min((int)(x), m-2));

          dst.at<float>(rhoMax - r, theta-minAngle) += (float)(xLow)*static_cast<float>(src.at<T>(y + n, (int)(x) + m))
                            + (float)(xUp)*static_cast<float>(src.at<T>(y + n, (int)(x) + m + 1));
        }
      }
    }

    // Range: [46, 90]
    else if( theta <= 89 ) {
      for(int r = 1; r <= rhoMax; ++r) {
        float rho = (float)(r - rc);
        float b = rho/sinTh;

        int xMax = (std::min((int)((-n - b)/a), m-1)),
            xMin = (std::max((int)((n - b)/a), -m));

        for(int x = xMin; x <= xMax; ++x) {
          float y = a*x + b;
          int yFloor = static_cast<int>(std::floor(y));
          float yUp = y - yFloor;
          float yLow = 1.0f - yUp;
          y = (float)(yFloor);
          y = static_cast<float>(std::max((int)(y), -n));
          y = static_cast<float>(std::min((int)(y), n-2));

          dst.at<float>(rhoMax - r, theta-minAngle) += (float)(yLow)*static_cast<float>(src.at<T>((int)(y) + n, x + m))
                          + (float)(yUp)*static_cast<float>(src.at<T>((int)(y) + n + 1, x + m));
        }
      }
    }

    // theta = 90 degrees | Integrating along a horizontal line
    else if( theta == 90 ) {
    for(int r = 1; r <= rhoMax; ++r) {
      float rho = (float)(r - rc);
      float b = rho;

      int xMax = m - 1, xMin = -m;

      for(int x = xMin; x <= xMax; ++x) {
        float y = a*x + b;
        int yFloor = static_cast<int>(std::floor(y));
        float yUp = y - yFloor;
        float yLow = 1.0f - yUp;
        y = (float)(yFloor);
        y = static_cast<float>(std::max((int)(y), -n));
        y = static_cast<float>(std::min((int)(y), n-2));

        dst.at<float>(rhoMax - r, 90-minAngle) += (float)(yLow)*static_cast<float>(src.at<T>((int)(y) + n, x + m))
                          + (float)(yUp)*static_cast<float>(src.at<T>((int)(y) + n + 1, x + m));
      }
    }
  }

    // Range: [91, 135]
    else if( theta <= 135 ) {
      for(int r = 1; r <= rhoMax; ++r) {
        float rho = (float)(r - rc);
        float b = rho/sinTh;

        int xMax = std::min((int)((n - b)/a), m-1),
            xMin = std::max((int)((-n - b)/a), -m);

        for(int x = xMin; x <= xMax; ++x) {
          float y = a*x + b;
          int yFloor = static_cast<int>(std::floor(y));
          float yUp = y - yFloor;
          float yLow = 1.0f - yUp;
          y = (float)(yFloor);
          y = static_cast<float>(std::max((int)(y), -n));
          y = static_cast<float>(std::min((int)(y), n-2));

          dst.at<float>(rhoMax - r, theta-minAngle) += (float)(yLow)*static_cast<float>(src.at<T>((int)(y) + n, x + m))
                              + (float)(yUp)*static_cast<float>(src.at<T>((int)(y) + n + 1, x + m));
        }
      }
    }

    // Range: [136, 179]
    else if( theta <= 179 ) {
      for(int r = 1; r <= rhoMax; ++r) {
        float rho = (float)(r - rc);
        float b = rho/sinTh;

        int yMax = std::min((int)(a*m + b), n-1),
            yMin = std::max((int)(-a*m + b), -n);

        for(int y = yMin; y <= yMax; ++y) {
          float x = (y - b)/a;
          int xFloor = static_cast<int>(std::floor(x));
          float xUp = x - xFloor;
          float xLow = 1.0f - xUp;
          x = (float)(xFloor);
          x = static_cast<float>(std::max((int)(x), -m));
          x = static_cast<float>(std::min((int)(x), m-2));

          dst.at<float>(rhoMax - r, theta-minAngle) += (float)(xLow)*static_cast<float>(src.at<T>(y + n, (int)(x) + m))
                              + (float)(xUp)*static_cast<float>(src.at<T>(y + n, (int)(x) + m + 1));
        }
      }
    }

    // theta = 180 degrees | Integrating along a vertical line
    else if( theta == 180 ) {
      int rhoOffset = static_cast<int>((rhoMax - src.cols)/2);
      for(int x = 1; x <= src.cols; ++x) {
        int r = x + rhoOffset;
        for(int y = 0; y < src.rows; ++y) {
          dst.at<float>(r, 180-minAngle) += static_cast<float>(src.at<T>(y, x-1));
        }
      }
    }
}

/**
 * [setMinMaxAngle Sets the given minimum and maximum angles if they are smaller and higher than the current values respectively]
 * @param currentMinAngle [Current minimum angle]
 * @param currentMaxAngle [Current maximum angle]
 * @param minAngle [Given minimum angle to set as current minimum angle, referenced]
 * @param maxAngle [Given maximum angle to set as current maximum angle, referenced]
 */
static inline void setMinMaxAngle( int & currentMinAngle,
                                   int & currentMaxAngle,
                                   int minAngle,
                                   int maxAngle)
{
  if (minAngle < currentMinAngle)
  {
    currentMinAngle = minAngle;
  }
  if (maxAngle > currentMaxAngle)
  {
    currentMaxAngle = maxAngle;
  }
}

/**
 * [getAngleRange Computes minAngle and maxAngle for given angleRange - AR_x_y]
 * @param angleRange [Given flag for angle range - AR_x_y]
 * @param minAngle   [minAngle, referenced]
 * @param maxAngle   [maxAngle, referenced]
 */

static inline void getAngleRange( int angleRange,
                                  int & minAngle,
                                  int & maxAngle
                                ) {
  minAngle = 180;
  maxAngle = 0;
  if (angleRange & AR_1_45)
  {
    setMinMaxAngle(minAngle, maxAngle, 1, 45);
  }
  if (angleRange & AR_46_89)
  {
    setMinMaxAngle(minAngle, maxAngle, 46, 89);
  }
  if (angleRange & AR_90)
  {
    setMinMaxAngle(minAngle, maxAngle, 90, 90);
  }
  if (angleRange & AR_91_135)
  {
    setMinMaxAngle(minAngle, maxAngle, 91, 135);
  }
  if (angleRange & AR_136_179)
  {
    setMinMaxAngle(minAngle, maxAngle, 136, 179);
  }
  if (angleRange & AR_180)
  {
    setMinMaxAngle(minAngle, maxAngle, 180, 180);
  }
}

template <typename T, int D>
static void rT_T( Mat srcMat,
                  Mat & dstMat,
                  int minAngle,
                  int maxAngle
                ) {
  int rows = srcMat.rows, cols = srcMat.cols;
  int m = cols/2, n = rows/2;

  for(int i = minAngle; i <= maxAngle; ++i) {
    trans<T, D>(srcMat, dstMat, m, n, dstMat.rows - 1, i, minAngle);
  }
}

static void rT( Mat src,
                Mat & dst,
                int minAngle,
                int maxAngle
              ) {
  int const depth = src.depth();
  switch (depth) {
    case CV_8U:
      rT_T<uchar, CV_8UC1>(src, dst, minAngle, maxAngle);
      break;
    case CV_8S:
      rT_T<schar, CV_8SC1>(src, dst, minAngle, maxAngle);
      break;
    case CV_16U:
      rT_T<ushort, CV_16UC1>(src, dst, minAngle, maxAngle);
      break;
    case CV_16S:
      rT_T<short, CV_16SC1>(src, dst, minAngle, maxAngle);
      break;
    case CV_32S:
      rT_T<int, CV_32SC1>(src, dst, minAngle, maxAngle);
      break;
    case CV_32F:
      rT_T<float, CV_32FC1>(src, dst, minAngle, maxAngle);
      break;
    case CV_64F:
      rT_T<double, CV_64FC1>(src, dst, minAngle, maxAngle);
      break;
    default:
      CV_Error_(CV_StsNotImplemented, ("Unknown depth %d", depth));
      break;
  }
}

/**
 * Creates the destination image with appropriate size, channels and depth
 */
static void createDstRadonMat( OutputArray dst,
                               InputArray src,
                               int depth,
                               int minAngle,
                               int maxAngle
                             ) {
  int const rows = src.size().height;
  int const cols = src.size().width;
  int const channels = src.channels();

  if(depth == -1) {
    depth = src.depth();
  }

  int rhoMax = static_cast<int>(std::ceil(std::sqrt(rows*rows + cols*cols)));
  int lenTheta = maxAngle - minAngle + 1;

  dst.create(Size(lenTheta, rhoMax + 1), CV_MAKETYPE(depth, channels));
}

/**
 * [radonTransform Computes the Radon transform of the given image]
 * @param src        [Input source image]
 * @param dst        [Output image]
 * @param angleRange [One of the enums, RadonAngleRange, defaults to 63 ( 1 to 180 degrees)]
 * @param operation  [One of the enums, RadonOp, defaults to RT_SUM ( SUM OPERATION )]
 * @param opDst      [Output image of the operation perfomed, defaults to noArray()]
 */
void radonTransform( InputArray src,
                     OutputArray dst,
                     int angleRange,
                     int operation,
                     OutputArray opDst
                   ) {
  /**
   * Asserting proper width, height and channels.
   * Input should be a one channel image.
   */
  CV_Assert(src.size().width > 0 && src.size().height > 0);
  CV_Assert(src.channels() == 1);

  Mat srcMat = src.getMat().clone();

  int minAngle, maxAngle;
  getAngleRange(angleRange, minAngle, maxAngle);
  createDstRadonMat(dst, src, CV_32F, minAngle, maxAngle);

  Mat dstMat = dst.getMat();
  rT(srcMat, dstMat, minAngle, maxAngle);

  // Operating the Radon transform
  if( opDst.needed() ) {
    opDst.create(Size(maxAngle - minAngle + 1, 1), CV_32F);
    Mat opDstMat = opDst.getMat();

    switch(operation) {
      case RT_MIN:
        reduce(dstMat, opDstMat, 0, REDUCE_MIN);
        break;
      case RT_AVE:
        reduce(dstMat, opDstMat, 0, REDUCE_AVG);
        break;
      case RT_MAX:
        reduce(dstMat, opDstMat, 0, REDUCE_MAX);
        break;
      case RT_SUM:
      default:
        reduce(dstMat, opDstMat, 0, REDUCE_SUM);
        break;
    }
  }
}

#ifdef __cplusplus
  }
}
#endif
