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
    CV_WRAP bool empty() const CV_OVERRIDE { return _refImgs.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _refImgs.clear(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates quality
    @param refImgs input image(s) to use as the source for comparison
    */
    CV_WRAP static Ptr<QualityMSE> create(InputArrayOfArrays refImgs);

    /**
    @brief static method for computing quality
    @param refImgs reference image(s)
    @param cmpImgs comparison image(s)
    @param qualityMaps output quality map(s), or cv::noArray()
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to potentially max float (worst)
    */
    CV_WRAP static cv::Scalar compute( InputArrayOfArrays refImgs, InputArrayOfArrays cmpImgs, OutputArrayOfArrays qualityMaps );

protected:

    /** @brief Reference images, converted to internal mat type */
    std::vector<QualityBase::_quality_map_type> _refImgs;

    /**
    @brief Constructor
    @param refImgs vector of reference images, converted to internal type
    */
    QualityMSE(std::vector<QualityBase::_quality_map_type> refImgs)
        : _refImgs(std::move(refImgs))
    {}

};  // QualityMSE
}   // quality
}   // cv
#endif