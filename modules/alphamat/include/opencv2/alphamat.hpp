/*
 By downloading, copying, installing or using the software you agree to this
 license. If you do not agree to this license, do not download, install,
 copy or use the software.
                License Agreement
        For Open Source Computer Vision Library
                (3-clause BSD License)
 Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the contributors
 may be used to endorse or promote products derived from this software
 without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are
 disclaimed. In no event shall copyright holders or contributors be liable for
 any direct, indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
 */

/** Information Flow algorithm implementaton for alphamatting */

#ifndef _OPENCV_ALPHAMAT_HPP_
#define _OPENCV_ALPHAMAT_HPP_

/**
 * @defgroup alphamat Alpha Matting
 * This module is dedicated to compute alpha matting of images, given the input image and an input trimap.
 * The samples directory includes easy examples of how to use the module.
 *
 * The implementation is based on Designing Effective Inter-Pixel Information Flow for Natural Image Matting by Yağız Aksoy, Tunç Ozan Aydın and Marc Pollefeys, CVPR 2019.
 *
 * This module has been originally developed by Muskaan Kularia and Sunita Nayak as a project
 * for Google Summer of Code 2019 (GSoC 19).
 *
 */

namespace cv{ namespace alphamat{

CV_EXPORTS_W void infoFlow(Mat& image, Mat& tmap, Mat& result);

}}
#endif
