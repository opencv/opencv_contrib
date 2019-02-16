// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/quality/qualitygmsd.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/imgproc.hpp"  // blur, resize
#include "opencv2/quality/quality_utils.hpp"

namespace
{
    using namespace cv;
    using namespace cv::quality;

    using _mat_type = cv::UMat;// match QualityGMSD::_mat_data::mat_type
    using _quality_map_type = _mat_type;

    template <typename SrcMat, typename DstMat>
    void filter_2D(const SrcMat& src, DstMat& dst, cv::InputArray kernel, cv::Point anchor, double delta, int border_type )
    {
        cv::filter2D(src, dst, src.depth(), kernel, anchor, delta, border_type);
    }

    // At the time of this writing (OpenCV 4.0.1) cv::Filter2D with OpenCL+UMat/32F suffers from precision loss large enough
    //  to warrant conversion prior to application of Filter2D
    template <typename DstMat>
    void filter_2D( const UMat& src, DstMat& dst, cv::InputArray kernel, cv::Point anchor, double delta, int border_type )
    {
        if ( !cv::ocl::useOpenCL() || src.depth() == CV_64F)   // nothing more to do
            return filter_2D<UMat, DstMat>(src, dst, kernel, anchor, delta, border_type);

        auto dst_type = dst.type() == 0 ? src.type() : dst.type();

        // UMat conversion to 64F
        UMat src_converted = {};
        src.convertTo(src_converted, CV_64F);
        dst.convertTo(dst, CV_64F);

        filter_2D<UMat, DstMat>(src_converted, dst, kernel, anchor, delta, border_type);
        dst.convertTo(dst, dst_type);
    }

    // conv2, based on https://stackoverflow.com/a/12540358
    enum ConvolutionType {
        /* Return the full convolution, including border */
        CONVOLUTION_FULL,

        /* Return only the part that corresponds to the original image */
        CONVOLUTION_SAME,

        /* Return only the submatrix containing elements that were not influenced by the border */
        CONVOLUTION_VALID
    };

    template <typename MatSrc, typename MatDst, typename TKernel>
    void conv2(const MatSrc& img, MatDst& dest, const TKernel& kernel, ConvolutionType type ) {
        auto source = img;
        TKernel kernel_flipped = {};
        cv::flip(kernel, kernel_flipped, -1);

        if (CONVOLUTION_FULL == type) {
            source = MatSrc();
            const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
            cv::copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2,
                (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
        }

        cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);

        // cv::filter2D(source, dest, img.depth(), kernel_flipped, anchor, 0, BORDER_CONSTANT );
        filter_2D(source, dest, kernel_flipped, anchor, 0, BORDER_CONSTANT);

        if (CONVOLUTION_VALID == type) {
            dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2)
                .rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
        }
    }
}   // ns

// construct mat_data from _mat_type
QualityGMSD::_mat_data::_mat_data(const QualityGMSD::_mat_data::mat_type& mat)
{
    CV_Assert(!mat.empty());

    // 2x2 avg kernel
    _mat_type
        tmp1 = {}
        , tmp = {}
    ;

    cv::blur(mat, tmp1, cv::Size(2, 2), cv::Point(0, 0), BORDER_CONSTANT);

    // 2x2 downsample
    // bug/hack:
    //  modules\core\src\matrix.cpp:169: error: (-215:Assertion failed) u->refcount == 0 in function 'cv::StdMatAllocator::deallocate'
    //  when src==dst and using UMat, useOpenCL=false
    //  workaround:  use 2 temp vars instead of 1 so that src != dst
    //  todo:  fix after https://github.com/opencv/opencv/issues/13577 solved
    cv::resize(tmp1, tmp, cv::Size(), .5, .5, INTER_NEAREST);

    // prewitt conv2
    static const cv::Matx33d
        prewitt_y = { 1. / 3., 1. / 3., 1. / 3., 0., 0., 0., -1. / 3., -1. / 3., -1. / 3. }
        , prewitt_x = { 1. / 3., 0., -1. / 3., 1. / 3., 0., -1. / 3.,1. / 3., 0., -1. / 3. }
    ;

    // prewitt y on tmp ==> this->gradient_map
    ::conv2(tmp, this->gradient_map, prewitt_y, ::ConvolutionType::CONVOLUTION_SAME);

    // prewitt x on tmp ==> tmp
    ::conv2(tmp, tmp, prewitt_x, ::ConvolutionType::CONVOLUTION_SAME);

    // calc gradient map, sqrt( px ^ 2 + py ^ 2 )
    cv::multiply(this->gradient_map, this->gradient_map, this->gradient_map);   // square gradient map
    cv::multiply(tmp, tmp, tmp);    // square temp
    cv::add(this->gradient_map, tmp, this->gradient_map);   // add together
    cv::sqrt(this->gradient_map, this->gradient_map);// get sqrt

    // calc gradient map squared
    this->gradient_map_squared = this->gradient_map.mul(this->gradient_map);
}

// static
Ptr<QualityGMSD> QualityGMSD::create(InputArrayOfArrays refImgs)
{
    return Ptr<QualityGMSD>(new QualityGMSD( _mat_data::create( refImgs )));
}

// static
cv::Scalar QualityGMSD::compute(InputArrayOfArrays refImgs, InputArrayOfArrays cmpImgs, OutputArrayOfArrays qualityMaps)
{
    auto ref = _mat_data::create( refImgs );
    auto cmp = _mat_data::create( cmpImgs );

    return _mat_data::compute(ref, cmp, qualityMaps);
}

cv::Scalar QualityGMSD::compute(InputArrayOfArrays cmpImgs)
{
    auto cmp = _mat_data::create(cmpImgs);
    return _mat_data::compute(this->_refImgData, cmp, this->_qualityMaps);
}

// static, converts mat/umat to vector of mat_data
std::vector<QualityGMSD::_mat_data> QualityGMSD::_mat_data::create(InputArrayOfArrays arr)
{
    std::vector<_mat_data> result = {};
    auto mats = quality_utils::expand_mats<_mat_type>(arr);
    result.reserve(mats.size());
    for (auto& mat : mats)
        result.emplace_back(mat);
    return result;
}

// computes gmsd and quality map for single frame
std::pair<cv::Scalar, _quality_map_type> QualityGMSD::_mat_data::compute(const QualityGMSD::_mat_data& lhs, const QualityGMSD::_mat_data& rhs)
{
    static const double T = 170.;
    std::pair<cv::Scalar, _quality_map_type> result;

    // compute quality_map = (2 * gm1 .* gm2 + T) ./ (gm1 .^2 + gm2 .^2 + T);
    _mat_type num
        , denom
        , qm
        ;

    cv::multiply(lhs.gradient_map, rhs.gradient_map, num);
    cv::multiply(num, 2., num);
    cv::add(num, T, num);

    cv::add(lhs.gradient_map_squared, rhs.gradient_map_squared, denom);
    cv::add(denom, T, denom);

    cv::divide(num, denom, qm);

    cv::meanStdDev(qm, cv::noArray(), result.first);
    result.second = std::move(qm);

    return result;
}   // compute

// static, computes mse and quality maps for multiple frames
cv::Scalar QualityGMSD::_mat_data::compute(const std::vector<QualityGMSD::_mat_data>& lhs, const std::vector<QualityGMSD::_mat_data>& rhs, OutputArrayOfArrays qualityMaps)
{
    CV_Assert(lhs.size() > 0);
    CV_Assert(lhs.size() == rhs.size());

    cv::Scalar result = {};
    std::vector<_quality_map_type> quality_maps = {};
    const auto sz = lhs.size();

    for (unsigned i = 0; i < sz; ++i)
    {
        CV_Assert(!lhs.empty() && !rhs.empty());

        auto cmp = compute(lhs[i], rhs[i]); // differs slightly when using umat vs mat

        cv::add(result, cmp.first, result);     // result += cmp.first

        if (qualityMaps.needed())
            quality_maps.emplace_back(std::move(cmp.second));
    }

    if (qualityMaps.needed())
    {
        auto qMaps = InputArray(quality_maps);
        qualityMaps.create(qMaps.size(), qMaps.type());
        qualityMaps.assign(quality_maps);
    }

    if (sz > 1)
        result /= (cv::Scalar::value_type)sz;  // average result

    return result;
}