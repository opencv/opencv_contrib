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
// Copyright (C) 2014, Beat Kueng (beat-kueng@gmx.net), Lukas Vogel, Morten Lysgaard
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
#include <cmath>

namespace cv {
namespace ximgproc {

void niBlackThreshold( InputArray _src, OutputArray _dst, double maxValue,
        int type, int blockSize, double k, int binarizationMethod, double r)
{
    // Input grayscale image
    Mat src = _src.getMat();
    CV_Assert(src.channels() == 1);
    CV_Assert(blockSize % 2 == 1 && blockSize > 1);
    if (binarizationMethod == BINARIZATION_SAUVOLA) {
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(r != 0);
    }
    type &= THRESH_MASK;

    // Compute local threshold (T = mean + k * stddev)
    // using mean and standard deviation in the neighborhood of each pixel
    // (intermediate calculations are done with floating-point precision)
    Mat thresh;
    {
        // note that: Var[X] = E[X^2] - E[X]^2
        Mat mean, sqmean, variance, stddev, sqrtVarianceMeanSum;
        double srcMin, stddevMax;
        boxFilter(src, mean, CV_32F, Size(blockSize, blockSize),
                Point(-1,-1), true, BORDER_REPLICATE);
        sqrBoxFilter(src, sqmean, CV_32F, Size(blockSize, blockSize),
                Point(-1,-1), true, BORDER_REPLICATE);
        variance = sqmean - mean.mul(mean);
        sqrt(variance, stddev);
        switch (binarizationMethod)
        {
        case BINARIZATION_NIBLACK:
            thresh = mean + stddev * static_cast<float>(k);
            break;
        case BINARIZATION_SAUVOLA:
            thresh = mean.mul(1. + static_cast<float>(k) * (stddev / r - 1.));
            break;
        case BINARIZATION_WOLF:
            minMaxIdx(src, &srcMin);
            minMaxIdx(stddev, NULL, &stddevMax);
            thresh = mean - static_cast<float>(k) * (mean - srcMin - stddev.mul(mean - srcMin) / stddevMax);
            break;
        case BINARIZATION_NICK:
            sqrt(variance + sqmean, sqrtVarianceMeanSum);
            thresh = mean + static_cast<float>(k) * sqrtVarianceMeanSum;
            break;
        default:
            CV_Error( Error::StsBadArg, "Unknown binarization method" );
            break;
        }
        thresh.convertTo(thresh, src.depth());
    }

    // Prepare output image
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    CV_Assert(src.data != dst.data);  // no inplace processing

    // Apply thresholding: ( pixel > threshold ) ? foreground : background
    Mat mask;
    switch (type)
    {
    case THRESH_BINARY:      // dst = (src > thresh) ? maxval : 0
    case THRESH_BINARY_INV:  // dst = (src > thresh) ? 0 : maxval
        compare(src, thresh, mask, (type == THRESH_BINARY ? CMP_GT : CMP_LE));
        dst.setTo(0);
        dst.setTo(maxValue, mask);
        break;
    case THRESH_TRUNC:       // dst = (src > thresh) ? thresh : src
        compare(src, thresh, mask, CMP_GT);
        src.copyTo(dst);
        thresh.copyTo(dst, mask);
        break;
    case THRESH_TOZERO:      // dst = (src > thresh) ? src : 0
    case THRESH_TOZERO_INV:  // dst = (src > thresh) ? 0 : src
        compare(src, thresh, mask, (type == THRESH_TOZERO ? CMP_GT : CMP_LE));
        dst.setTo(0);
        src.copyTo(dst, mask);
        break;
    default:
        CV_Error( Error::StsBadArg, "Unknown threshold type" );
        break;
    }
}

} // namespace ximgproc
} // namespace cv
