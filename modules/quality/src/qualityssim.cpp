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

    // Cached 1D Gaussian kernel for separable filtering (CV_32F for better performance)
    static Mat getGaussianKernel1D()
    {
        static Mat kernel = cv::getGaussianKernel(SSIM_KERNEL_SIZE, SSIM_SIGMA, CV_32F);
        return kernel;
    }

    // Optimized blur using separable filter with cached kernel
    // Performance: O(2k) vs O(k^2) for 2D Gaussian blur
    void blur(InputArray src, OutputArray dst)
    {
        Mat kernel = getGaussianKernel1D();
        cv::sepFilter2D(src, dst, CV_32F, kernel, kernel);
    }

#ifdef HAVE_OPENCL
    // OpenCL kernel for SSIM map computation
    // Fuses the final SSIM formula into a single GPU kernel
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

    // CPU implementation for SSIM computation
    static cv::Scalar cpu_ssim_compute(
        const _mat_type& mu1, const _mat_type& mu2,
        const _mat_type& mu1_sq, const _mat_type& mu2_sq,
        const _mat_type& sigma1_sq, const _mat_type& sigma2_sq,
        const _mat_type& sigma12,
        _mat_type* out_ssim_map)
    {
        _mat_type mu1_mu2, ssim_map, temp;

        cv::multiply(mu1, mu2, mu1_mu2);

        // Compute numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        cv::addWeighted(mu1_mu2, 2.0, mu1_mu2, 0.0, C1, ssim_map);
        cv::addWeighted(sigma12, 2.0, sigma12, 0.0, C2, temp);
        cv::multiply(ssim_map, temp, ssim_map);

        // Compute denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        cv::addWeighted(mu1_sq, 1.0, mu2_sq, 1.0, C1, mu1_mu2);
        cv::addWeighted(sigma1_sq, 1.0, sigma2_sq, 1.0, C2, temp);
        cv::multiply(mu1_mu2, temp, temp);

        // quality map = numerator / denominator
        cv::divide(ssim_map, temp, ssim_map);

        cv::Scalar result = cv::mean(ssim_map);

        if (out_ssim_map)
            *out_ssim_map = std::move(ssim_map);

        return result;
    }

}   // ns

// Construct _mat_data from a matrix
// Precomputes values needed for SSIM: I, I^2, mu, mu^2, sigma^2
QualitySSIM::_mat_data::_mat_data(const _mat_type& mat)
{
    // Convert to CV_32F for all computations (faster than CV_64F on GPU)
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
        true  // always compute map for instance method (stored in _qualityMap)
    );

    OutputArray(this->_qualityMap).assign(result.second);
    return result.first;
}

// static. Computes SSIM and optionally quality map
// Optimized with:
// 1. Separable Gaussian filter (O(2k) vs O(k^2))
// 2. CV_32F precision (faster on GPU)
// 3. OpenCL kernel for fused SSIM computation
std::pair<cv::Scalar, _mat_type> QualitySSIM::_mat_data::compute(const _mat_data& lhs, const _mat_data& rhs, bool need_quality_map)
{
    mat_type sigma12;

    // Compute sigma12 = blur(I1 * I2) - mu1 * mu2
    mat_type I1_I2;
    cv::multiply(lhs.I, rhs.I, I1_I2);
    ::blur(I1_I2, sigma12);
    mat_type mu1_mu2;
    cv::multiply(lhs.mu, rhs.mu, mu1_mu2);
    cv::subtract(sigma12, mu1_mu2, sigma12);

#ifdef HAVE_OPENCL
    if (cv::ocl::isOpenCLActivated())
    {
        mat_type ssim_map;
        if (ocl_ssim_map(lhs.mu, rhs.mu, lhs.mu_2, rhs.mu_2,
                        lhs.sigma_2, rhs.sigma_2, sigma12, ssim_map))
        {
            cv::Scalar result = cv::mean(ssim_map);
            if (!need_quality_map)
                ssim_map.release();  // Free memory if not needed
            return { result, std::move(ssim_map) };
        }
    }
#endif

    // CPU fallback
    mat_type ssim_map;
    cv::Scalar result = cpu_ssim_compute(
        lhs.mu, rhs.mu, lhs.mu_2, rhs.mu_2,
        lhs.sigma_2, rhs.sigma_2, sigma12,
        need_quality_map ? &ssim_map : nullptr
    );

    return { result, std::move(ssim_map) };
}
