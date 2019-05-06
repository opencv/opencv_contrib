// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYSSIM_HPP
#define OPENCV_QUALITY_QUALITYSSIM_HPP

#include "qualitybase.hpp"

namespace cv
{
namespace quality
{

/**
@brief Full reference structural similarity algorithm  https://en.wikipedia.org/wiki/Structural_similarity
*/
class CV_EXPORTS_W QualitySSIM
    : public QualityBase {
public:

    /**
    @brief Computes SSIM
    @param cmp Comparison image
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (worst) to 1 (best)
    */
    CV_WRAP cv::Scalar compute( InputArray cmp ) CV_OVERRIDE;

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _refImgData.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _refImgData = _mat_data(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates quality
    @param ref input image to use as the reference image for comparison
    */
    CV_WRAP static Ptr<QualitySSIM> create( InputArray ref );

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image
    @param qualityMap output quality map, or cv::noArray()
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (worst) to 1 (best)
    */
    CV_WRAP static cv::Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap );

protected:

    // holds computed values for a mat
    struct _mat_data
    {
        // internal mat type
        using mat_type = QualityBase::_mat_type;

        mat_type
            I
            , I_2
            , mu
            , mu_2
            , sigma_2
            ;

        // allow default construction
        _mat_data() = default;

        // construct from mat_type
        _mat_data(const mat_type&);

        // construct from inputarray
        _mat_data(InputArray);

        // return flag if this is empty
        bool empty() const { return I.empty() && I_2.empty() && mu.empty() && mu_2.empty() && sigma_2.empty(); }

        // computes ssim and quality map for single frame
        static std::pair<cv::Scalar, mat_type> compute(const _mat_data& lhs, const _mat_data& rhs);

    };  // mat_data

    /** @brief Reference image data */
    _mat_data _refImgData;

    /**
    @brief Constructor
    @param refImgData reference image, converted to internal type
    */
    QualitySSIM( _mat_data refImgData )
        : _refImgData( std::move(refImgData) )
    {}

};  // QualitySSIM
}   // quality
}   // cv
#endif