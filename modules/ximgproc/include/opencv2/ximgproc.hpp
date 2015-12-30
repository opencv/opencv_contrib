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

#ifndef __OPENCV_XIMGPROC_HPP__
#define __OPENCV_XIMGPROC_HPP__

#include "ximgproc/edge_filter.hpp"
#include "ximgproc/disparity_filter.hpp"
#include "ximgproc/sparse_match_interpolator.hpp"
#include "ximgproc/structured_edge_detection.hpp"
#include "ximgproc/seeds.hpp"
#include "ximgproc/segmentation.hpp"
#include "ximgproc/fast_hough_transform.hpp"
#include "ximgproc/estimated_covariance.hpp"
#include "ximgproc/slic.hpp"
#include "ximgproc/lsc.hpp"

/** @defgroup ximgproc Extended Image Processing
  @{
    @defgroup ximgproc_edge Structured forests for fast edge detection

This module contains implementations of modern structured edge detection algorithms, i.e. algorithms
which somehow takes into account pixel affinities in natural images.

    @defgroup ximgproc_filters Filters

    @defgroup ximgproc_superpixel Superpixels

    @defgroup ximgproc_segmentation Image segmentation
  @}
*/

namespace cv {
namespace ximgproc {
    CV_EXPORTS_W
    void niBlackThreshold( InputArray _src, OutputArray _dst, double maxValue,
            int type, int blockSize, double delta );

} // namespace ximgproc
} //namespace cv

#endif
