// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/quality/qualitymae.hpp"
#include "opencv2/quality/quality_utils.hpp"

namespace cv
{

namespace quality
{

using namespace quality_utils;


// Static
Ptr<QualityMAE> QualityMAE::create(InputArray ref, int statsProc)
{
    return Ptr<QualityMAE>(new QualityMAE(quality_utils::expand_mat<_mat_type>(ref), statsProc));
}

// Static
Scalar QualityMAE::compute(InputArray ref, InputArray cmp, OutputArray qualityMap, int statsProc)
{
    CV_Assert_3(ref.channels() <= 4,
                cmp.channels() <= 4,
                (statsProc == MAE_MAX) || (statsProc == MAE_MEAN) );

    _mat_type err;
    int wdepth = std::max(std::max(ref.depth(), cmp.depth()), CV_32F);
    int cn = ref.channels();
    int wtype = CV_MAKETYPE(wdepth, cn);

    absdiff(extract_mat<_mat_type>(ref, wtype), extract_mat<_mat_type>(cmp, wtype), err);

    if(qualityMap.needed())
        qualityMap.assign(statsProc == MAE_MAX ? err : err.clone());

    if(statsProc == MAE_MEAN)
    {
        return mean(err);
    }

    Scalar scores;
    _mat_type tmp = err.reshape(err.channels(), 1);

    reduce(tmp, tmp, 1, REDUCE_MAX, wdepth);

    tmp.convertTo(Mat(tmp.size(), CV_64FC(cn), scores.val), CV_64F);

    return scores;
}

// Not static
Scalar QualityMAE::compute( InputArray cmpImg )
{
    CV_Assert(cmpImg.isMat() || cmpImg.isUMat() || cmpImg.isMatx());

    if(cmpImg.empty())
        return Scalar();

    // If the input is a set of images.
    _mat_type cmp = extract_mat<_mat_type>(cmpImg, std::max(cmpImg.depth(), CV_32F));

    return QualityMAE::compute(this->_ref, cmp, this->_qualityMap, this->_flag);
}

QualityMAE::QualityMAE(QualityBase::_mat_type ref, int flag)
    : _ref(std::move(ref)),
      _flag(flag)
{}

} // quality

} // cv
