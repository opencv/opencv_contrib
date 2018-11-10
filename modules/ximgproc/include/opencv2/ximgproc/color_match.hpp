// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_COLOR_MATCH_HPP__
#define __OPENCV_COLOR_MATCH_HPP__

#include <opencv2/core.hpp>

namespace cv {
namespace ximgproc {

//! @addtogroup ximgproc_filters
//! @{

/**
* @brief   creates a quaternion image.
*
* @param   img         Source 8-bit, 32-bit or 64-bit image, with 3-channel image.
* @param   qimg        result CV_64FC4 a quaternion image( 4 chanels zero channel and B,G,R).
*/
CV_EXPORTS_W void createQuaternionImage(InputArray img, OutputArray qimg);

/**
* @brief   calculates conjugate of a quaternion image.
*
* @param   qimg         quaternion image.
* @param   qcimg        conjugate of qimg
*/
CV_EXPORTS_W void qconj(InputArray qimg, OutputArray qcimg);
/**
* @brief   divides each element by its modulus.
*
* @param   qimg         quaternion image.
* @param   qnimg        conjugate of qimg
*/
CV_EXPORTS_W void qunitary(InputArray qimg, OutputArray qnimg);
/**
* @brief   Calculates the per-element quaternion product of two arrays
*
* @param   src1         quaternion image.
* @param   src2         quaternion image.
* @param   dst        product dst(I)=src1(I) . src2(I)
*/
CV_EXPORTS_W void qmultiply(InputArray  	src1, InputArray  	src2, OutputArray  	dst);
/**
* @brief    Performs a forward or inverse Discrete quaternion Fourier transform of a 2D quaternion array.
*
* @param   img        quaternion image.
* @param   qimg       quaternion image in dual space.
* @param   flags      quaternion image in dual space. only DFT_INVERSE flags is supported
* @param   sideLeft   true the hypercomplex exponential is to be multiplied on the left (false on the right ).
*/
CV_EXPORTS_W void qdft(InputArray img, OutputArray qimg, int  	flags, bool sideLeft);
/**
* @brief    Compares a color template against overlapped color image regions.
*
* @param   img        Image where the search is running. It must be 3 channels image
* @param   templ       Searched template. It must be not greater than the source image and have 3 channels
* @param   result     Map of comparison results. It must be single-channel 64-bit floating-point
*/
CV_EXPORTS_W void colorMatchTemplate(InputArray img, InputArray templ, OutputArray result);

}
}
#endif
