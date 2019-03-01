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

    // computes mse and quality maps for multiple frames
    cv::Scalar compute(const std::vector<mse_mat_type>& lhs, const std::vector<mse_mat_type>& rhs, OutputArrayOfArrays qualityMaps )
    {
        CV_Assert(lhs.size() > 0);
        CV_Assert(lhs.size() == rhs.size());

        cv::Scalar result = {};
        std::vector<_quality_map_type> quality_maps = {};
        const auto sz = lhs.size();

        for (unsigned i = 0; i < sz; ++i)
        {
            CV_Assert(!lhs.empty() && !rhs.empty());

            auto cmp = compute(lhs[i], rhs[i]);
            cv::add(result, cmp.first, result);

            if ( qualityMaps.needed() )
                quality_maps.emplace_back(std::move(cmp.second));
        }

        if (qualityMaps.needed())
        {
            auto qMaps = InputArray(quality_maps);
            qualityMaps.create(qMaps.size(), qMaps.type());
            qualityMaps.assign(quality_maps);
        }

        if (sz > 1)
            result /= (cv::Scalar::value_type)sz;   // average result

        return result;
    }
}

// static
Ptr<QualityMSE> QualityMSE::create( InputArrayOfArrays refImgs )
{
    return Ptr<QualityMSE>(new QualityMSE(quality_utils::expand_mats<mse_mat_type>(refImgs)));
}

// static
cv::Scalar QualityMSE::compute( InputArrayOfArrays refImgs, InputArrayOfArrays cmpImgs, OutputArrayOfArrays qualityMaps )
{
    auto ref = quality_utils::expand_mats<mse_mat_type>(refImgs);
    auto cmp = quality_utils::expand_mats<mse_mat_type>(cmpImgs);

    return ::compute(ref, cmp, qualityMaps);
}

cv::Scalar QualityMSE::compute( InputArrayOfArrays cmpImgs )
{
    auto cmp = quality_utils::expand_mats<mse_mat_type>(cmpImgs);
    return ::compute( this->_refImgs, cmp, this->_qualityMaps);
}