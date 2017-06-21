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
 //                        (3-clause BSD License)
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



#ifndef __OPENCV_OBSTRUCTION_FREE_CPP__
#define __OPENCV_OBSTRUCTION_FREE_CPP__

#include <vector>
#include <iostream>

#include <opencv2/xphoto.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{
namespace xphoto
{
void obstructionFree(const std::vector <Mat> &srcImgs, Mat &dst, Mat &mask){
    size_t frameNumber = srcImgs.size();
    size_t reference_number = (frameNumber-1)/2;;
    dst.create( srcImgs[reference_number].size(), srcImgs[reference_number].type() );
    mask = Mat::zeros(dst.rows,dst.cols,CV_8UC1);


        /////////construct image pyramids//////
    int pyramid_level=3;
    std::vector<Mat> video_coarseLevel;
    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        Mat temp, temp_gray;
        temp=srcImgs[frame_i].clone();
        cvtColor(temp, temp_gray, COLOR_RGB2GRAY);
        for (int i=0; i<pyramid_level; i++){
            pyrDown( temp_gray, temp_gray );
        }
        video_coarseLevel.push_back(temp_gray.clone());
    }
}
}
}
#endif // __OPENCV_OBSTRUCTION_FREE_CPP__
