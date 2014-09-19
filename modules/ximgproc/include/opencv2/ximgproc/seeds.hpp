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

#ifndef __OPENCV_SEEDS_HPP__
#define __OPENCV_SEEDS_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{


//! Superpixel implementation: "SEEDS: Superpixels Extracted via Energy-Driven Sampling", IJCV 2014
class CV_EXPORTS_W SuperpixelSEEDS : public Algorithm
{
public:

    /*! get the actual number of superpixels */
    CV_WRAP virtual int getNumberOfSuperpixels() = 0;

    /*!
     * calculate the segmentation on a given image. To get the result use getLabels()
     * @param img            input image. supported formats: CV_8U, CV_16U, CV_32F
     *                       image size & number of channels must match with the
     *                       initialized image size & channels.
     * @param num_iterations number of pixel level iterations. higher number
     *                       improves the result
     */
    CV_WRAP virtual void iterate(InputArray img, int num_iterations=4) = 0;

    /*!
     * retrieve the segmentation results.
     * @param labels_out     Return: A CV_32UC1 integer array containing the labels
     *                       labels are in the range [0, getNumberOfSuperpixels()]
     */
    CV_WRAP virtual void getLabels(OutputArray labels_out) = 0;

    /*!
     * get an image mask with the contour of the superpixels. useful for test output.
     * @param image          Return: CV_8UC1 image mask where -1 is a superpixel border
     *                       pixel and 0 an interior pixel.
     * @param thick_line     if false, border is only one pixel wide, otherwise
     *                       all border pixels are masked
     */
    CV_WRAP virtual void getLabelContourMask(OutputArray image, bool thick_line = false) = 0;

    virtual ~SuperpixelSEEDS() {}
};

/*! Creates a SuperpixelSEEDS object.
 * @param image_width       image width
 * @param image_height      image height
 * @param image_channels    number of channels the image has
 * @param num_superpixels   desired number of superpixels. Note that the actual
 *                          number can be smaller due to further restrictions.
 *                          use getNumberOfSuperpixels to get the actual number.
 * @param num_levels        number of block levels: the more levels, the more
 *                          accurate is the segmentation, but needs more memory
 *                          and CPU time.
 * @param histogram_bins    number of histogram bins.
 * @param prior             enable 3x3 shape smoothing term if >0. a larger value
 *                          leads to smoother shapes.
 *                          range: [0, 5]
 * @param double_step       if true, iterate each block level twice for higher
 *                          accuracy.
 */
CV_EXPORTS_W Ptr<SuperpixelSEEDS> createSuperpixelSEEDS(
    int image_width, int image_height, int image_channels,
    int num_superpixels, int num_levels, int prior = 2,
    int histogram_bins=5, bool double_step = false);


}
}
#endif
#endif
