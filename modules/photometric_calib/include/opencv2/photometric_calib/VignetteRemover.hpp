// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_VIGNETTEREMOVER_HPP
#define _OPENCV_VIGNETTEREMOVER_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace photometric_calib {

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
     * @param pcalibPath the path of pcalib file
     * @param w_ the width of input image
     * @param h_ the height of input image
     */
    VignetteRemover(const std::string &vignettePath, const std::string &pcalibPath, int w_, int h_);

    ~VignetteRemover();

    /*!
     * @brief get vignetting-removed image in form of cv::Mat.
     * @param oriImMat the image to be calibrated.
     */
    Mat getUnVignetteImageMat(Mat oriImMat);

    /*!
     * @brief get vignetting-removed image in form of std::vector<float>.
     * @param oriImMat the image to be calibrated.
     * @param outImVec the vignetting-removed image vector.
     */
    void getUnVignetteImageVec(Mat oriImMat, std::vector<float> &outImVec);

private:
    float *vignetteMap;
    float *vignetteMapInv;
    std::string _pcalibPath;
    int w, h;
    bool validVignette;
};

} // namespace photometric_calib
} // namespace cv

#endif //_OPENCV_VIGNETTEREMOVER_HPP
