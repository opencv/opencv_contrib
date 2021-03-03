// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/quality/qualityssim.hpp"
#include "opencv2/imgproc.hpp"  // GaussianBlur
#include "opencv2/quality/quality_utils.hpp"

namespace
{
    using namespace cv;
    using namespace cv::quality;

    using _mat_type = UMat;
    using _quality_map_type = _mat_type;

    // SSIM blur function
    _mat_type blur(const _mat_type& mat)
    {
        _mat_type result = {};
        cv::GaussianBlur( mat, result, cv::Size(11, 11), 1.5 );
        return result;
    }
}   // ns

QualitySSIM::_mat_data::_mat_data( const _mat_type& mat )
{
    this->I = mat;
    cv::multiply(this->I, this->I, this->I_2);
    this->mu = ::blur(this->I);
    cv::multiply(this->mu, this->mu, this->mu_2);
    this->sigma_2 = ::blur(this->I_2);    // blur the squared img, subtract blurred_squared
    cv::subtract(this->sigma_2, this->mu_2, this->sigma_2);
}

QualitySSIM::_mat_data::_mat_data(InputArray arr )
    : _mat_data( quality_utils::expand_mat<mat_type>(arr) )    // delegate
{}

// static
Ptr<QualitySSIM> QualitySSIM::create( InputArray ref )
{
    return Ptr<QualitySSIM>(new QualitySSIM( _mat_data( ref )));
}

// static
cv::Scalar QualitySSIM::compute( InputArray ref, InputArray cmp, OutputArray qualityMap )
{
    auto result = _mat_data::compute( _mat_data(ref), _mat_data(cmp) );

    if (qualityMap.needed())
        qualityMap.assign(result.second);

    return result.first;
}

cv::Scalar QualitySSIM::compute( InputArray cmp )
{
    auto result = _mat_data::compute(
        this->_refImgData
        , _mat_data(cmp)
    );

    OutputArray(this->_qualityMap).assign(result.second);
    return result.first;
}

// static.  computes ssim and quality map for single frame
// based on https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html
std::pair<cv::Scalar, _mat_type> QualitySSIM::_mat_data::compute(const _mat_data& lhs, const _mat_data& rhs)
{
    const double
        C1 = 6.5025
        , C2 = 58.5225
        ;

    mat_type
        I1_I2
        , mu1_mu2
        , t1
        , t2
        , t3
        , sigma12
        ;

    cv::multiply(lhs.I, rhs.I, I1_I2);
    cv::multiply(lhs.mu, rhs.mu, mu1_mu2);
    cv::subtract(::blur(I1_I2), mu1_mu2, sigma12);

    // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    cv::multiply(mu1_mu2, 2., t1);
    cv::add(t1, C1, t1);// t1 += C1

    cv::multiply(sigma12, 2., t2);
    cv::add(t2, C2, t2);// t2 += C2

    // t3 = t1 * t2
    cv::multiply(t1, t2, t3);

    // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    cv::add(lhs.mu_2, rhs.mu_2, t1);
    cv::add(t1, C1, t1);

    cv::add(lhs.sigma_2, rhs.sigma_2, t2);
    cv::add(t2, C2, t2);

    // t1 *= t2
    cv::multiply(t1, t2, t1);

    // quality map: t3 /= t1
    cv::divide(t3, t1, t3);

    return {
        cv::mean(t3)
        , std::move(t3)
    };
}   // compute