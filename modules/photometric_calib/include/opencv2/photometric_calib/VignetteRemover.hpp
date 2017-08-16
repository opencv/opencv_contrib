// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_VIGNETTEREMOVER_HPP
#define _OPENCV_VIGNETTEREMOVER_HPP

#include "opencv2/core.hpp"

namespace cv { namespace photometric_calib {

//! @addtogroup photometric_calib
//! @{

/*!
 * @brief Class for removing the vignetting artifact when provided with vignetting file.
 *
 */

class CV_EXPORTS VignetteRemover
{
public:
    /*!
     * @brief Constructor
     * @param vignettePath the path of vignetting file
     * @param w_ the width of input image
     * @param h_ the height of input image
     */
    VignetteRemover(const std::string &vignettePath, int w_, int h_);
    ~VignetteRemover();

    /*!
     * @brief get vignetting-removed image in form of cv::Mat.
     * @param unGammaImVec the irradiance image.
     * @return
     */
    Mat getUnVignetteImageMat(std::vector<float> &unGammaImVec);

    /*!
     * @brief get vignetting-removed image in form of std::vector<float>.
     * @param unGammaImVec the irradiance image.
     * @param outImVec the vignetting-removed image vector.
     */
    void getUnVignetteImageVec(const std::vector<float> &unGammaImVec, std::vector<float> &outImVec);

private:
    float* vignetteMap;
    float* vignetteMapInv;
    int w,h;
    bool validVignette;
};

}}


#endif //_OPENCV_VIGNETTEREMOVER_HPP
