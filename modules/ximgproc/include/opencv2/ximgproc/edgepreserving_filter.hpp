// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_EDGEPRESERVINGFILTER_HPP__
#define __OPENCV_EDGEPRESERVINGFILTER_HPP__

#include <opencv2/core.hpp>

namespace cv { namespace ximgproc {

//! @addtogroup ximgproc
//! @{

    /**
    * @brief Smoothes an image using the Edge-Preserving filter.
    *
    * The function smoothes Gaussian noise as well as salt & pepper noise.
    * For more details about this implementation, please see
    * [ReiWoe18]  Reich, S. and Wörgötter, F. and Dellen, B. (2018). A Real-Time Edge-Preserving Denoising Filter. Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP): Visapp, 85-94, 4. DOI: 10.5220/0006509000850094.
    *
    * @param src Source 8-bit 3-channel image.
    * @param dst Destination image of the same size and type as src.
    * @param d Diameter of each pixel neighborhood that is used during filtering. Must be greater or equal 3.
    * @param threshold Threshold, which distinguishes between noise, outliers, and data.
    */
    CV_EXPORTS_W void edgePreservingFilter( InputArray src, OutputArray dst, int d, double threshold );

//! @}

}} // namespace

#endif
