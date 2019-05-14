// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYMSE_HPP
#define OPENCV_QUALITY_QUALITYMSE_HPP

#include "qualitybase.hpp"

namespace cv
{
namespace quality
{

/**
@brief Full reference mean square error algorithm  https://en.wikipedia.org/wiki/Mean_squared_error
*/
class CV_EXPORTS_W QualityMSE : public QualityBase {
public:

    /** @brief Computes MSE for reference images supplied in class constructor and provided comparison images
    @param cmpImgs Comparison image(s)
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to potentially max float (worst)
    */
    CV_WRAP cv::Scalar compute( InputArrayOfArrays cmpImgs ) CV_OVERRIDE;

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _ref.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _ref = _mat_type(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates quality
    @param ref input image to use as the reference for comparison
    */
    CV_WRAP static Ptr<QualityMSE> create(InputArray ref);

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image=
    @param qualityMap output quality map, or cv::noArray()
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to max float (worst)
    */
    CV_WRAP static cv::Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap );

protected:

    /** @brief Reference image, converted to internal mat type */
    QualityBase::_mat_type _ref;

    /**
    @brief Constructor
    @param ref reference image, converted to internal type
    */
    QualityMSE(QualityBase::_mat_type ref)
        : _ref(std::move(ref))
    {}

};  // QualityMSE
}   // quality
}   // cv
#endif