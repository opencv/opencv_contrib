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
        int type, int blockSize, double delta )
{
    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
    Size size = src.size();

    _dst.create( size, src.type() );
    Mat dst = _dst.getMat();

    if( maxValue < 0 )
    {
        dst = Scalar(0);
        return;
    }

    // Calculate and store the mean and mean of squares in the neighborhood
    // of each pixel and store them in Mat mean and sqmean.
    Mat_<float> mean(size), sqmean(size);

    if( src.data != dst.data )
        mean = dst;

    boxFilter( src, mean, CV_64F, Size(blockSize, blockSize),
            Point(-1,-1), true, BORDER_REPLICATE );
    sqrBoxFilter( src, sqmean, CV_64F, Size(blockSize, blockSize),
            Point(-1,-1), true, BORDER_REPLICATE );

    // Compute (k * standard deviation) in the neighborhood of each pixel
    // and store in Mat stddev. Also threshold the values in the src matrix to compute dst matrix.
    Mat_<float> stddev(size);
    int i, j, threshold;
    uchar imaxval = saturate_cast<uchar>(maxValue);
    for(i = 0; i < size.height; ++i)
    {
        for(j = 0; j < size.width; ++j)
        {
            stddev.at<float>(i, j) = saturate_cast<float>(delta) * cvRound( sqrt(sqmean.at<float>(i, j) -
                        mean.at<float>(i, j)*mean.at<float>(i, j)) );
            threshold = cvRound(mean.at<float>(i, j) + stddev.at<float>(i, j));
            if(src.at<uchar>(i, j) > threshold)
                dst.at<uchar>(i, j) = (type == THRESH_BINARY) ? imaxval : 0;
            else
                dst.at<uchar>(i, j) = (type == THRESH_BINARY) ? 0 : imaxval;
        }
    }

}

} // namespace ximgproc
} //namespace cv
