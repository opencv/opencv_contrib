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
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_IMG_HASH_BASE_HPP__
#define __OPENCV_IMG_HASH_BASE_HPP__

#include "opencv2/core.hpp"

namespace cv
{

    namespace img_hash
    {
        //! @addtogroup ihash
        //! @{
        /**@brief The base class for image hash algorithms
         */
        class ImgHashBase : public Algorithm
        {
        public:
            /** @brief Computes hash of the input image
            @param inputArr input image want to compute hash value
            @param outputArr hash of the image
            */
            CV_EXPORTS virtual void compute(cv::InputArray inputArr,
                                            cv::OutputArray outputArr) = 0;

            /** @brief Compare the hash value between inOne and inTwo
            @param hashOne Hash value one
            @param hashTwo Hash value two
            @return value indicate similarity between inOne and inTwo, the meaning
            of the value vary from algorithms to algorithms
            */
            CV_EXPORTS virtual double compare(cv::InputArray hashOne,
                                              cv::InputArray hashTwo) const = 0;
        };
        //! @}
    }//ihash
}//cv

#endif // __OPENCV_IMG_HASH_BASE_HPP__
