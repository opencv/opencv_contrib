// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_GAMMAREMOVER_HPP
#define _OPENCV_GAMMAREMOVER_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace photometric_calib {

//! @addtogroup photometric_calib
//! @{

/*!
 * @brief Class for removing the camera response function (mostly gamma function) when provided with pcalib file.
 *
 */

class CV_EXPORTS GammaRemover
{
public:
    /*!
     * @brief Constructor
     * @param gammaPath the path of pcalib file of which the format should be .yaml or .yml
     * @param w_ the width of input image
     * @param h_ the height of input image
     */
    GammaRemover(const std::string &gammaPath, int w_, int h_);

    /*!
     * @brief get irradiance image in the form of cv::Mat. Convenient for display, etc.
     * @param inputIm
     * @return
     */
    Mat getUnGammaImageMat(Mat inputIm);

    /*!
     * @brief get irradiance image in the form of std::vector<float>. Convenient for optimization or SLAM.
     * @param inputIm
     * @param outImVec
     */
    void getUnGammaImageVec(Mat inputIm, std::vector<float> &outImVec);

    /*!
     * @brief get gamma function.
     * @return
     */
    inline float *getG()
    {
        if (!validGamma)
        { return 0; }
        else
        { return G; }
    };

    /*!
     * @brief get inverse gamma function
     * @return
     */
    inline float *getGInv()
    {
        if (!validGamma)
        { return 0; }
        else
        { return GInv; }
    };

private:
    float G[256];
    float GInv[256];
    int w, h;
    bool validGamma;
};

} // namespace photometric_calib
} // namespace cv

#endif //_OPENCV__GAMMAREMOVER_HPP
