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
    @param cmpImgs Comparison images
    @returns Per-channel GMSD
    */
    CV_WRAP cv::Scalar compute(InputArrayOfArrays cmpImgs) CV_OVERRIDE;

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _refImgData.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _refImgData.clear(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates image quality
    @param refImgs input image(s) to use as the source for comparison
    */
    CV_WRAP static Ptr<QualityGMSD> create(InputArrayOfArrays refImgs);

    /**
    @brief static method for computing quality
    @param refImgs reference image(s)
    @param cmpImgs comparison image(s)
    @param qualityMaps output quality map(s), or cv::noArray()
    @returns cv::Scalar with per-channel quality value.  Values range from 0 (worst) to 1 (best)
    */
    CV_WRAP static cv::Scalar compute(InputArrayOfArrays refImgs, InputArrayOfArrays cmpImgs, OutputArrayOfArrays qualityMaps);

protected:

    // holds computed values for an input mat
    struct _mat_data
    {
        using mat_type = QualityBase::_quality_map_type;

        mat_type
            gradient_map
            , gradient_map_squared
            ;

        _mat_data(const mat_type&);

        // converts mat/umat to vector of mat_data
        static std::vector<_mat_data> create(InputArrayOfArrays arr);

        // compute for a single frame
        static std::pair<cv::Scalar, mat_type> compute(const _mat_data& lhs, const _mat_data& rhs);

        // compute for vector of inputs
        static cv::Scalar compute(const std::vector<_mat_data>& lhs, const std::vector<_mat_data>& rhs, OutputArrayOfArrays qualityMaps);

    };  // mat_data

    /** @brief Reference image data */
    std::vector<_mat_data> _refImgData;

    /**
    @brief Constructor
    @param refImgData vector of reference images, converted to internal type
    */
    QualityGMSD(std::vector<_mat_data> refImgData)
        : _refImgData(std::move(refImgData))
    {}

};  // QualityGMSD
}   // quality
}   // cv
#endif