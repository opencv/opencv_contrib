// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_MCC_UTILS_HPP__
#define __OPENCV_MCC_UTILS_HPP__

#include <opencv2/core.hpp>

namespace cv {
namespace ccm {

CV_EXPORTS_W double gammaCorrection_(const double& element, const double& gamma);

/** @brief gamma correction ,see ColorSpace.pdf for details.
    @param src the input array,type of Mat.
    @param gamma a constant for gamma correction.
 */
CV_EXPORTS_W Mat gammaCorrection(const Mat& src, const double& gamma);

/** @brief maskCopyTo a function to delete unsatisfied elementwise.
    @param src the input array, type of Mat.
    @param mask operation mask that used to choose satisfided elementwise.
 */
CV_EXPORTS_W Mat maskCopyTo(const Mat& src, const Mat& mask);

/** @brief multiple the function used to compute an array with n channels
      mulipied by ccm.
    @param xyz the input array, type of Mat.
    @param ccm the ccm matrix to make color correction.
 */
CV_EXPORTS_W Mat multiple(const Mat& xyz, const Mat& ccm);

/** @brief multiple the function used to get the mask of saturated colors,
            colors between low and up will be choosed.
    @param src the input array, type of Mat.
    @param low  the threshold to choose saturated colors
    @param up  the threshold to choose saturated colors
*/
CV_EXPORTS_W Mat saturate(Mat& src, const double& low, const double& up);

/** @brief rgb2gray it is an approximation grayscale function for relative RGB
           color space, see Miscellaneous.pdf for details;
    @param  rgb the input array,type of Mat.
 */
CV_EXPORTS_W Mat rgb2gray(Mat rgb);

/** @brief function for elementWise operation
    @param src the input array, type of Mat
    @param lambda a for operation
 */
template<typename F>
Mat elementWise(const Mat& src, F&& lambda)
{
    Mat dst = src.clone();
    const int channel = src.channels();
    switch (channel)
    {
    case 1:
    {

        MatIterator_<double> it, end;
        for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it)
        {
            (*it) = lambda((*it));
        }
        break;
    }
    case 3:
    {
        MatIterator_<cv::Vec3d> it, end;
        for (it = dst.begin<cv::Vec3d>(), end = dst.end<cv::Vec3d>(); it != end; ++it)
        {
            for (int j = 0; j < 3; j++)
            {
                (*it)[j] = lambda((*it)[j]);
            }
        }
        break;
    }
    default:
        throw std::invalid_argument { "Wrong channel!" };
        break;
    }
    return dst;
}

/** @brief function for channel operation
      @param src the input array, type of Mat
      @param lambda the function for operation
*/
template<typename F>
Mat channelWise(const Mat& src, F&& lambda)
{
    Mat dst = src.clone();
    MatIterator_<cv::Vec3d> it, end;
    for (it = dst.begin<cv::Vec3d>(), end = dst.end<cv::Vec3d>(); it != end; ++it)
    {
        *it = lambda(*it);
    }
    return dst;
}

/** @brief function for distance operation.
    @param src the input array, type of Mat.
    @param ref another input array, type of Mat.
    @param lambda the computing method for distance .
 */
template<typename F>
Mat distanceWise(Mat& src, Mat& ref, F&& lambda)
{
    Mat dst = Mat(src.size(), CV_64FC1);
    MatIterator_<cv::Vec3d> it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>(),
        it_ref = ref.begin<cv::Vec3d>();
    MatIterator_<double> it_dst = dst.begin<double>();
    for (; it_src != end_src; ++it_src, ++it_ref, ++it_dst)
    {
        *it_dst = lambda(*it_src, *it_ref);
    }
    return dst;
}

CV_EXPORTS_W Mat multiple(const Mat& xyz, const Mat& ccm);

const static Mat m_gray = (Mat_<double>(3, 1) << 0.2126, 0.7152, 0.0722);

}  // namespace ccm
}  // namespace cv

#endif