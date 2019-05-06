// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/quality/qualitymse.hpp"
#include "opencv2/quality/quality_utils.hpp"

namespace
{
    using namespace cv;
    using namespace cv::quality;

    using mse_mat_type = UMat;
    using _quality_map_type = mse_mat_type;

    // computes mse and quality map for single frame
    std::pair<cv::Scalar, _quality_map_type> compute(const mse_mat_type& lhs, const mse_mat_type& rhs)
    {
        std::pair<cv::Scalar, _quality_map_type> result;

        cv::subtract( lhs, rhs, result.second );

        // cv::pow(diff, 2., diff);
        cv::multiply(result.second, result.second, result.second); // slightly faster than pow2

        result.first = cv::mean(result.second);

        return result;
    }
}

// static
Ptr<QualityMSE> QualityMSE::create( InputArray ref )
{
    return Ptr<QualityMSE>(new QualityMSE(quality_utils::expand_mat<mse_mat_type>(ref)));
}

// static
cv::Scalar QualityMSE::compute( InputArray ref_, InputArray cmp_, OutputArray qualityMap )
{
    auto ref = quality_utils::expand_mat<mse_mat_type>(ref_);
    auto cmp = quality_utils::expand_mat<mse_mat_type>(cmp_);

    auto result = ::compute(ref, cmp);

    if (qualityMap.needed())
        qualityMap.assign(result.second);

    return result.first;
}

cv::Scalar QualityMSE::compute( InputArray cmp_ )
{
    auto cmp = quality_utils::expand_mat<mse_mat_type>(cmp_);
    auto result = ::compute( this->_ref, cmp );
    OutputArray(this->_qualityMap).assign(result.second);
    return result.first;
}