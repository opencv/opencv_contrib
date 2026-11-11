// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_MARR_HILDRETH_HASH_HPP
#define OPENCV_MARR_HILDRETH_HASH_HPP

#include "img_hash_base.hpp"

namespace cv {
namespace img_hash {

//! @addtogroup img_hash
//! @{

/** @brief Computes OpenCV's Marr-Hildreth image hash.

OpenCV uses the kernel \f$(2-a)e^{a/2}\f$, where \f$a\f$ is the squared
distance from the kernel center in scaled coordinates. The Laplacian of
Gaussian described in @cite zauner2010implementation uses \f$e^{-a/2}\f$.
Hashes from OpenCV should only be compared with hashes produced using the same
implementation and kernel parameters.
*/
class CV_EXPORTS_W MarrHildrethHash : public ImgHashBase
{
public:
    /**
     * @brief self explain
     */
    CV_WRAP float getAlpha() const;

    /**
     * @brief self explain
     */
    CV_WRAP float getScale() const;

    /** @brief Sets the hash kernel parameters.
        @param alpha Base of the kernel scale factor.
        @param scale Exponent of the kernel scale factor.
    */
    CV_WRAP void setKernelParam(float alpha, float scale);

    /**
        @param alpha Base of the kernel scale factor (default 2).
        @param scale Exponent of the kernel scale factor (default 1).
    */
    CV_WRAP static Ptr<MarrHildrethHash> create(float alpha = 2.0f, float scale = 1.0f);
protected:
    MarrHildrethHash() {}
};

/** @brief Computes the Marr-Hildreth hash of the input image.
    @param inputArr Input image of type CV_8UC1, CV_8UC3, or CV_8UC4.
    @param outputArr A 1-by-72 matrix of type CV_8U containing the hash.
    @param alpha Base of the kernel scale factor (default 2).
    @param scale Exponent of the kernel scale factor (default 1).
    @sa MarrHildrethHash
*/
CV_EXPORTS_W void marrHildrethHash(cv::InputArray inputArr,
                                   cv::OutputArray outputArr,
                                   float alpha = 2.0f, float scale = 1.0f);

//! @}

}} // cv::img_hash::

#endif // OPENCV_MARR_HILDRETH_HASH_HPP
