// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_AUG_FUNCTIONAL_HPP
#define OPENCV_AUG_FUNCTIONAL_HPP
#include <opencv2/core.hpp>
#include <vector>

namespace cv {
    //! @addtogroup imgaug
    //! @{

    /** @brief Adjust the brightness of the given image.
     *
     * @param img Source image. This operation is inplace.
     * @param brightness_factor brightness factor which controls the brightness of the adjusted image.
     * Brightness factor should be >= 0. When brightness factor is larger than 1, the output image will be brighter than original.
     * When brightness factor is less than 1, the output image will be darker than original.
     */
    void adjustBrightness(Mat& img, double brightness_factor);

    /** @brief Adjust the contrast of the given image.
     *
     * @param img Source image. This operation is inplace.
     * @param contrast_factor contrast factor should be larger than 1. It controls the contrast of the adjusted image.
     */
    void adjustContrast(Mat& img, double contrast_factor);

    /** @brief Adjust the saturation of the given image.
     *
     * @param img Source image. This operation is inplace.
     * @param saturation_factor saturation factor should be larger than 1. It controls the saturation of the adjusted image.
     */
    void adjustSaturation(Mat& img, double saturation_factor);

    /** @brief Adjust the hue of the given image.
     *
     * @param img Source image. This operation is inplace.
     * @param hue_factor hue factor should be in range [-1, 1]. It controls the hue of the adjusted image.
     */
    void adjustHue(Mat& img, double hue_factor);

    //! @}
};

#endif