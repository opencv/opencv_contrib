/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  *Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_DERICHEFILTER_HPP__
#define __OPENCV_DERICHEFILTER_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv {
namespace ximgproc {

//! @addtogroup ximgproc_filters
//! @{

/**
* @brief   Applies Deriche filter to an image.
*
* For more details about this implementation, please see @cite zhang2014100+
*
* @param   joint       Joint 8-bit, 1-channel or 3-channel image.
* @param   src         Source 8-bit or floating-point, 1-channel or 3-channel image.
* @param   dst         Destination image.
* @param   r           Radius of filtering kernel, should be a positive integer.
* @param   sigma       Filter range standard deviation for the joint image.
* @param   weightType  weightType The type of weight definition, see WMFWeightType
* @param   mask        A 0-1 mask that has the same size with I. This mask is used to ignore the effect of some pixels. If the pixel value on mask is 0,
*                           the pixel will be ignored when maintaining the joint-histogram. This is useful for applications like optical flow occlusion handling.
*
* @sa medianBlur, jointBilateralFilter
*/
CV_EXPORTS UMat GradientDericheY(UMat op, double alphaDerive,double alphaMoyenne);
CV_EXPORTS UMat GradientDericheX(UMat op, double alphaDerive,double alphaMoyenne);

}
}
#endif
#endif
