// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYGMSD_HPP
#define OPENCV_QUALITY_QUALITYGMSD_HPP

#include "qualitybase.hpp"

namespace cv
{
namespace quality
{

/**
@brief Full reference GMSD algorithm
http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
*/
class CV_EXPORTS_W QualityGMSD
    : public QualityBase {
public:

    /**
    @brief Compute GMSD
    @param cmp comparison image
    @returns cv::Scalar with per-channel quality value.  Values range from 0 (worst) to 1 (best)
    */
    CV_WRAP cv::Scalar compute( InputArray cmp ) CV_OVERRIDE;

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _refImgData.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _refImgData = _mat_data(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates image quality
    @param ref reference image
    */
    CV_WRAP static Ptr<QualityGMSD> create( InputArray ref );

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image
    @param qualityMap output quality map, or cv::noArray()
    @returns cv::Scalar with per-channel quality value.  Values range from 0 (worst) to 1 (best)
    */
    CV_WRAP static cv::Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap );

protected:

    // holds computed values for a mat
    struct _mat_data
    {
        // internal mat type
        using mat_type = QualityBase::_mat_type;

        mat_type
            gradient_map
            , gradient_map_squared
            ;

        // allow default construction
        _mat_data() = default;

        // construct from mat_type
        _mat_data(const mat_type&);

        // construct from inputarray
        _mat_data(InputArray);

        // returns flag if empty
        bool empty() const { return this->gradient_map.empty() && this->gradient_map_squared.empty(); }

        // compute for a single frame
        static std::pair<cv::Scalar, mat_type> compute(const _mat_data& lhs, const _mat_data& rhs);

    };  // mat_data

    /** @brief Reference image data */
    _mat_data _refImgData;

    // internal constructor
    QualityGMSD(_mat_data refImgData)
        : _refImgData(std::move(refImgData))
    {}

};  // QualityGMSD
}   // quality
}   // cv
#endif