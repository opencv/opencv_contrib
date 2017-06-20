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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_PCALIB_CPP__
#define __OPENCV_PCALIB_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/pcalib.hpp"

namespace cv{ namespace pcalib{

using namespace std;

bool PhotometricCalibrator::validImgs(std::vector <Mat> &inputImgs, std::vector<double> &exposureTime)
{
    if(inputImgs.empty() || exposureTime.empty() || inputImgs.size() != exposureTime.size())
        return false;

    int width = 0, height = 0;
    for(size_t i = 0; i < inputImgs.size(); ++ i)
    {
        Mat img;
        img = inputImgs[i];
        if(img.type() != CV_8U)
        {
            cout<<"The type of the image should be CV_8U!"<<endl;
            return false;
        }
        if((width!=0 && width != img.cols) || img.cols==0)
        {
            cout<<"Width mismatch!"<<endl;
            return false;
        };
        if((height!=0 && height != img.rows) || img.rows==0)
        {
            cout<<"Height mismatch!"<<endl;
            return false;
        };
        width = img.cols;
        height = img.rows;
    }
    return true;
}

}} // namespace pcalib, cv

#endif // __OPENCV_PCALIB_CPP__
#endif // cplusplus