// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/quality/qualityssim.hpp"
#include "opencv2/quality/quality_utils.hpp"
#include "opencv2/imgproc.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_quality.hpp"
#endif

namespace
{
    using namespace cv;
    using namespace cv::quality;

    using _mat_type = UMat;
    using _quality_map_type = _mat_type;

    // SSIM constants
    static constexpr int SSIM_KERNEL_SIZE = 11;
    static constexpr double SSIM_SIGMA = 1.5;
    static constexpr double C1 = 6.5025;   // (0.01 * 255)^2
    static constexpr double C2 = 58.5225;  // (0.03 * 255)^2

    // 1D Gaussian kernel for the separable SSIM blur
    static const Mat& getGaussianKernel1D()
    {
        static const Mat kernel = cv::getGaussianKernel(SSIM_KERNEL_SIZE, SSIM_SIGMA, CV_32F);
        return kernel;
    }

    void blur(InputArray src, OutputArray dst)
    {
        const Mat& kernel = getGaussianKernel1D();
        cv::sepFilter2D(src, dst, CV_32F, kernel, kernel);
    }

#ifdef HAVE_OPENCL
    static bool ocl_ssim_map(
        const UMat& mu1, const UMat& mu2,
        const UMat& mu1_sq, const UMat& mu2_sq,
        const UMat& sigma1_sq, const UMat& sigma2_sq,
        const UMat& sigma12,
        UMat& ssim_map)
    {
        int cn = mu1.channels();
        if (cn < 1 || cn > 4)
            return false;

        ocl::Kernel k("ssim_map", ocl::quality::ssim_oclsrc,
                      format("-D cn=%d", cn));

        if (k.empty())
            return false;

        ssim_map.create(mu1.size(), CV_MAKETYPE(CV_32F, cn));

        k.args(
            ocl::KernelArg::ReadOnlyNoSize(mu1),
            ocl::KernelArg::ReadOnlyNoSize(mu2),
            ocl::KernelArg::ReadOnlyNoSize(mu1_sq),
            ocl::KernelArg::ReadOnlyNoSize(mu2_sq),
            ocl::KernelArg::ReadOnlyNoSize(sigma1_sq),
            ocl::KernelArg::ReadOnlyNoSize(sigma2_sq),
            ocl::KernelArg::ReadOnlyNoSize(sigma12),
            ocl::KernelArg::WriteOnly(ssim_map),
            static_cast<float>(C1), static_cast<float>(C2)
        );

        size_t globalsize[2] = { (size_t)mu1.cols, (size_t)mu1.rows };
        return k.run(2, globalsize, NULL, false);
    }
#endif

    // Final SSIM step (CPU), one pass over the maps:
    //   sigma12 = blur_i1i2 - mu1_mu2
    //   ssim = (2*mu1_mu2 + C1)*(2*sigma12 + C2) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    static cv::Scalar cpu_ssim_compute(
        const _mat_type& mu1_mu2_u, const _mat_type& blur_i1i2_u,
        const _mat_type& mu1_sq_u, const _mat_type& mu2_sq_u,
        const _mat_type& sigma1_sq_u, const _mat_type& sigma2_sq_u,
        _mat_type* out_ssim_map)
    {
        Mat mu1_mu2   = mu1_mu2_u.getMat(ACCESS_READ);
        Mat blur_i1i2 = blur_i1i2_u.getMat(ACCESS_READ);
        Mat mu1_sq    = mu1_sq_u.getMat(ACCESS_READ);
        Mat mu2_sq    = mu2_sq_u.getMat(ACCESS_READ);
        Mat sigma1_sq = sigma1_sq_u.getMat(ACCESS_READ);
        Mat sigma2_sq = sigma2_sq_u.getMat(ACCESS_READ);

        Mat ssim(mu1_mu2.size(), mu1_mu2.type());

        const float c1 = static_cast<float>(C1);
        const float c2 = static_cast<float>(C2);
        const int rows = mu1_mu2.rows;
        const int width = mu1_mu2.cols * mu1_mu2.channels();  // element-wise over all channels

        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
        {
            for (int y = range.start; y < range.end; ++y)
            {
                const float* mm = mu1_mu2.ptr<float>(y);
                const float* bi = blur_i1i2.ptr<float>(y);
                const float* p1 = mu1_sq.ptr<float>(y);
                const float* p2 = mu2_sq.ptr<float>(y);
                const float* s1 = sigma1_sq.ptr<float>(y);
                const float* s2 = sigma2_sq.ptr<float>(y);
                float* dst = ssim.ptr<float>(y);
                for (int x = 0; x < width; ++x)
                {
                    const float m = mm[x];
                    const float sigma12 = bi[x] - m;
                    const float num = (2.0f * m + c1) * (2.0f * sigma12 + c2);
                    const float den = (p1[x] + p2[x] + c1) * (s1[x] + s2[x] + c2);
                    dst[x] = num / den;
                }
            }
        });

        cv::Scalar result = cv::mean(ssim);

        if (out_ssim_map)
            ssim.copyTo(*out_ssim_map);

        return result;
    }

}   // ns

// Precompute the per-image terms: I, I^2, mu = blur(I), mu^2, sigma^2 = blur(I^2) - mu^2
QualitySSIM::_mat_data::_mat_data(const _mat_type& mat)
{
    mat.convertTo(this->I, CV_32F);
    cv::multiply(this->I, this->I, this->I_2);
    ::blur(this->I, this->mu);
    cv::multiply(this->mu, this->mu, this->mu_2);
    ::blur(this->I_2, this->sigma_2);
    cv::subtract(this->sigma_2, this->mu_2, this->sigma_2);
}

QualitySSIM::_mat_data::_mat_data(InputArray arr)
    : _mat_data(quality_utils::expand_mat<mat_type>(arr))    // delegate
{}

// static
Ptr<QualitySSIM> QualitySSIM::create(InputArray ref)
{
    return Ptr<QualitySSIM>(new QualitySSIM(_mat_data(ref)));
}

// static
cv::Scalar QualitySSIM::compute(InputArray ref, InputArray cmp, OutputArray qualityMap)
{
    auto result = _mat_data::compute(
        _mat_data(ref),
        _mat_data(cmp),
        qualityMap.needed()
    );

    if (qualityMap.needed())
        qualityMap.assign(result.second);

    return result.first;
}

cv::Scalar QualitySSIM::compute(InputArray cmp)
{
    auto result = _mat_data::compute(
        this->_refImgData,
        _mat_data(cmp),
        true  // map is stored in _qualityMap
    );

    OutputArray(this->_qualityMap).assign(result.second);
    return result.first;
}

// static. Computes the mean SSIM and, optionally, the quality map.
std::pair<cv::Scalar, _mat_type> QualitySSIM::_mat_data::compute(const _mat_data& lhs, const _mat_data& rhs, bool need_quality_map)
{
    mat_type I1_I2;
    cv::multiply(lhs.I, rhs.I, I1_I2);
    mat_type blur_i1i2;
    ::blur(I1_I2, blur_i1i2);
    mat_type mu1_mu2;
    cv::multiply(lhs.mu, rhs.mu, mu1_mu2);

#ifdef HAVE_OPENCL
    if (cv::ocl::isOpenCLActivated())
    {
        mat_type sigma12;
        cv::subtract(blur_i1i2, mu1_mu2, sigma12);
        mat_type ssim_map;
        if (ocl_ssim_map(lhs.mu, rhs.mu, lhs.mu_2, rhs.mu_2,
                        lhs.sigma_2, rhs.sigma_2, sigma12, ssim_map))
        {
            cv::Scalar result = cv::mean(ssim_map);
            if (!need_quality_map)
                ssim_map.release();
            return { result, std::move(ssim_map) };
        }
    }
#endif

    mat_type ssim_map;
    cv::Scalar result = cpu_ssim_compute(
        mu1_mu2, blur_i1i2, lhs.mu_2, rhs.mu_2,
        lhs.sigma_2, rhs.sigma_2,
        need_quality_map ? &ssim_map : nullptr
    );

    return { result, std::move(ssim_map) };
}
