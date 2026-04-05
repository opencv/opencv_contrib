/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || !defined (HAVE_OPENCV_CUDAARITHM) || defined (CUDA_DISABLER)

Ptr<cuda::TemplateMatching> cv::cuda::createTemplateMatching(int, int, Size) { throw_no_cuda(); return Ptr<cuda::TemplateMatching>(); }

void cv::cuda::matchTemplate(InputArray, InputArray, OutputArray, int, InputArray, Stream&) { throw_no_cuda(); }

#else

namespace cv { namespace cuda { namespace device
{
    namespace match_template
    {
        void matchTemplateNaive_CCORR_8U(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);
        void matchTemplateNaive_CCORR_32F(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);

        void matchTemplateNaive_SQDIFF_8U(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);
        void matchTemplateNaive_SQDIFF_32F(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);

        void matchTemplatePrepared_SQDIFF_8U(int w, int h, const PtrStepSz<double> image_sqsum, double templ_sqsum, PtrStepSzf result,
            int cn, cudaStream_t stream);

        void matchTemplatePrepared_SQDIFF_NORMED_8U(int w, int h, const PtrStepSz<double> image_sqsum, double templ_sqsum, PtrStepSzf result,
            int cn, cudaStream_t stream);

        void matchTemplatePrepared_CCOFF_8U(int w, int h, const PtrStepSz<int> image_sum, int templ_sum, PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_8UC2(
            int w, int h,
            const PtrStepSz<int> image_sum_r,
            const PtrStepSz<int> image_sum_g,
            int templ_sum_r,
            int templ_sum_g,
            PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_8UC3(
                int w, int h,
                const PtrStepSz<int> image_sum_r,
                const PtrStepSz<int> image_sum_g,
                const PtrStepSz<int> image_sum_b,
                int templ_sum_r,
                int templ_sum_g,
                int templ_sum_b,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_8UC4(
                int w, int h,
                const PtrStepSz<int> image_sum_r,
                const PtrStepSz<int> image_sum_g,
                const PtrStepSz<int> image_sum_b,
                const PtrStepSz<int> image_sum_a,
                int templ_sum_r,
                int templ_sum_g,
                int templ_sum_b,
                int templ_sum_a,
                PtrStepSzf result, cudaStream_t stream);


        void matchTemplatePrepared_CCOFF_NORMED_8U(
                int w, int h, const PtrStepSz<int> image_sum,
                const PtrStepSz<double> image_sqsum,
                int templ_sum, double templ_sqsum,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_NORMED_8UC2(
                int w, int h,
                const PtrStepSz<int> image_sum_r, const PtrStepSz<double> image_sqsum_r,
                const PtrStepSz<int> image_sum_g, const PtrStepSz<double> image_sqsum_g,
                int templ_sum_r, double templ_sqsum_r,
                int templ_sum_g, double templ_sqsum_g,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_NORMED_8UC3(
                int w, int h,
                const PtrStepSz<int> image_sum_r, const PtrStepSz<double> image_sqsum_r,
                const PtrStepSz<int> image_sum_g, const PtrStepSz<double> image_sqsum_g,
                const PtrStepSz<int> image_sum_b, const PtrStepSz<double> image_sqsum_b,
                int templ_sum_r, double templ_sqsum_r,
                int templ_sum_g, double templ_sqsum_g,
                int templ_sum_b, double templ_sqsum_b,
                PtrStepSzf result, cudaStream_t stream);
        void matchTemplatePrepared_CCOFF_NORMED_8UC4(
                int w, int h,
                const PtrStepSz<int> image_sum_r, const PtrStepSz<double> image_sqsum_r,
                const PtrStepSz<int> image_sum_g, const PtrStepSz<double> image_sqsum_g,
                const PtrStepSz<int> image_sum_b, const PtrStepSz<double> image_sqsum_b,
                const PtrStepSz<int> image_sum_a, const PtrStepSz<double> image_sqsum_a,
                int templ_sum_r, double templ_sqsum_r,
                int templ_sum_g, double templ_sqsum_g,
                int templ_sum_b, double templ_sqsum_b,
                int templ_sum_a, double templ_sqsum_a,
                PtrStepSzf result, cudaStream_t stream);

        void normalize_8U(int w, int h, const PtrStepSz<double> image_sqsum,
                          double templ_sqsum, PtrStepSzf result, int cn, cudaStream_t stream);

        void extractFirstChannel_32F(const PtrStepSzb image, PtrStepSzf result, int cn, cudaStream_t stream);
    }
}}}

namespace
{
    // Evaluates optimal template's area threshold. If
    // template's area is less  than the threshold, we use naive match
    // template version, otherwise FFT-based (if available)
    int getTemplateThreshold(int method, int depth)
    {
        switch (method)
        {
        case TM_CCORR:
            if (depth == CV_32F) return 250;
            if (depth == CV_8U) return 300;
            break;

        case TM_SQDIFF:
            if (depth == CV_8U) return 300;
            break;
        }

        CV_Error(Error::StsBadArg, "unsupported match template mode");
        return 0;
    }

    // Masked matchTemplate (cv::cuda::matchTemplate with mask): scratch storage and per-call context.
    struct MatchTemplateMaskGpuBuffers
    {
        GpuMat work;
        GpuMat img2;
        GpuMat mask2;
        GpuMat temp_result;
        GpuMat temp_x;
        GpuMat tpl_masked;
        GpuMat diff_templ;
        GpuMat templx_mask;
        GpuMat img_mask_corr;
        GpuMat img_mask2_corr;
        GpuMat temp_res;
        GpuMat temp_res1;
        GpuMat temp_res2;
        GpuMat temp_mid;
        GpuMat two_m;
        GpuMat norm_imgx;
        GpuMat temp_for_norm;
        std::vector<GpuMat> spl, tplp, dstp;
    };

    struct MatchTemplateMaskCtx
    {
        GpuMat& img_f;
        GpuMat& tpl_f;
        GpuMat& mask_f;
        GpuMat& result;
        Size corrSize;
        int cn;
        Stream& stream;
        Ptr<Convolution>& conv;
        MatchTemplateMaskGpuBuffers& buf;
    };

    ///////////////////////////////////////////////////////////////
    // CCORR_32F

    class Match_CCORR_32F : public TemplateMatching
    {
    public:
        explicit Match_CCORR_32F(Size user_block_size);

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        Ptr<cuda::Convolution> conv_;
        GpuMat result_;
    };

    Match_CCORR_32F::Match_CCORR_32F(Size user_block_size)
    {
        conv_ = cuda::createConvolution(user_block_size);
    }

    void Match_CCORR_32F::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& _stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_32F );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
        GpuMat result = _result.getGpuMat();

        if (templ.size().area() < getTemplateThreshold(TM_CCORR, CV_32F))
        {
            matchTemplateNaive_CCORR_32F(image, templ, result, image.channels(), stream);
            return;
        }

        if (image.channels() == 1)
        {
            conv_->convolve(image.reshape(1), templ.reshape(1), result, true, _stream);
        }
        else
        {
            conv_->convolve(image.reshape(1), templ.reshape(1), result_, true, _stream);
            extractFirstChannel_32F(result_, result, image.channels(), stream);
        }
    }

    ///////////////////////////////////////////////////////////////
    // CCORR_8U

    class Match_CCORR_8U : public TemplateMatching
    {
    public:
        explicit Match_CCORR_8U(Size user_block_size) : match32F_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat imagef_, templf_;
        Match_CCORR_32F match32F_;
    };

    void Match_CCORR_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        if (templ.size().area() < getTemplateThreshold(TM_CCORR, CV_8U))
        {
            _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
            GpuMat result = _result.getGpuMat();

            matchTemplateNaive_CCORR_8U(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        image.convertTo(imagef_, CV_32F, stream);
        templ.convertTo(templf_, CV_32F, stream);

        match32F_.match(imagef_, templf_, _result, stream);
    }

    ///////////////////////////////////////////////////////////////
    // CCORR_NORMED_8U

    class Match_CCORR_NORMED_8U : public TemplateMatching
    {
    public:
        explicit Match_CCORR_NORMED_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        Match_CCORR_8U match_CCORR_;
        GpuMat image_sqsums_;
    };

    void Match_CCORR_NORMED_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        cuda::sqrIntegral(image.reshape(1), image_sqsums_, stream);

        double templ_sqsum = cuda::sqrSum(templ.reshape(1))[0];

        normalize_8U(templ.cols, templ.rows, image_sqsums_, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // SQDIFF_32F

    class Match_SQDIFF_32F : public TemplateMatching
    {
    public:
        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());
    };

    void Match_SQDIFF_32F::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_32F );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
        GpuMat result = _result.getGpuMat();

        matchTemplateNaive_SQDIFF_32F(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // SQDIFF_8U

    class Match_SQDIFF_8U : public TemplateMatching
    {
    public:
        explicit Match_SQDIFF_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat image_sqsums_;
        Match_CCORR_8U match_CCORR_;
    };

    void Match_SQDIFF_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        if (templ.size().area() < getTemplateThreshold(TM_SQDIFF, CV_8U))
        {
            _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);
            GpuMat result = _result.getGpuMat();

            matchTemplateNaive_SQDIFF_8U(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        cuda::sqrIntegral(image.reshape(1), image_sqsums_, stream);

        double templ_sqsum = cuda::sqrSum(templ.reshape(1))[0];

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        matchTemplatePrepared_SQDIFF_8U(templ.cols, templ.rows, image_sqsums_, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // SQDIFF_NORMED_8U

    class Match_SQDIFF_NORMED_8U : public TemplateMatching
    {
    public:
        explicit Match_SQDIFF_NORMED_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat image_sqsums_;
        Match_CCORR_8U match_CCORR_;
    };

    void Match_SQDIFF_NORMED_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        cuda::sqrIntegral(image.reshape(1), image_sqsums_, stream);

        double templ_sqsum = cuda::sqrSum(templ.reshape(1))[0];

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        matchTemplatePrepared_SQDIFF_NORMED_8U(templ.cols, templ.rows, image_sqsums_, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }

    ///////////////////////////////////////////////////////////////
    // CCOFF_8U

    class Match_CCOEFF_8U : public TemplateMatching
    {
    public:
        explicit Match_CCOEFF_8U(Size user_block_size) : match_CCORR_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        std::vector<GpuMat> images_;
        std::vector<GpuMat> image_sums_;
        Match_CCORR_8U match_CCORR_;
    };

    void Match_CCOEFF_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        match_CCORR_.match(image, templ, _result, stream);
        GpuMat result = _result.getGpuMat();

        if (image.channels() == 1)
        {
            image_sums_.resize(1);
            cuda::integral(image, image_sums_[0], stream);

            int templ_sum = (int) cuda::sum(templ)[0];

            matchTemplatePrepared_CCOFF_8U(templ.cols, templ.rows, image_sums_[0], templ_sum, result, StreamAccessor::getStream(stream));
        }
        else
        {
            cuda::split(image, images_);

            image_sums_.resize(images_.size());
            for (int i = 0; i < image.channels(); ++i)
                cuda::integral(images_[i], image_sums_[i], stream);

            Scalar templ_sum = cuda::sum(templ);

            switch (image.channels())
            {
            case 2:
                matchTemplatePrepared_CCOFF_8UC2(
                        templ.cols, templ.rows, image_sums_[0], image_sums_[1],
                        (int) templ_sum[0], (int) templ_sum[1],
                        result, StreamAccessor::getStream(stream));
                break;
            case 3:
                matchTemplatePrepared_CCOFF_8UC3(
                        templ.cols, templ.rows, image_sums_[0], image_sums_[1], image_sums_[2],
                        (int) templ_sum[0], (int) templ_sum[1], (int) templ_sum[2],
                        result, StreamAccessor::getStream(stream));
                break;
            case 4:
                matchTemplatePrepared_CCOFF_8UC4(
                        templ.cols, templ.rows, image_sums_[0], image_sums_[1], image_sums_[2], image_sums_[3],
                        (int) templ_sum[0], (int) templ_sum[1], (int) templ_sum[2], (int) templ_sum[3],
                        result, StreamAccessor::getStream(stream));
                break;
            default:
                CV_Error(Error::StsBadArg, "unsupported number of channels");
            }
        }
    }

    ///////////////////////////////////////////////////////////////
    // CCOFF_NORMED_8U

    class Match_CCOEFF_NORMED_8U : public TemplateMatching
    {
    public:
        explicit Match_CCOEFF_NORMED_8U(Size user_block_size) : match_CCORR_32F_(user_block_size)
        {
        }

        void match(InputArray image, InputArray templ, OutputArray result, Stream& stream = Stream::Null());

    private:
        GpuMat imagef_, templf_;
        Match_CCORR_32F match_CCORR_32F_;
        std::vector<GpuMat> images_;
        std::vector<GpuMat> image_sums_;
        std::vector<GpuMat> image_sqsums_;
    };

    void Match_CCOEFF_NORMED_8U::match(InputArray _image, InputArray _templ, OutputArray _result, Stream& stream)
    {
        using namespace cv::cuda::device::match_template;

        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( image.depth() == CV_8U );
        CV_Assert( image.type() == templ.type() );
        CV_Assert( image.cols >= templ.cols && image.rows >= templ.rows );

        image.convertTo(imagef_, CV_32F, stream);
        templ.convertTo(templf_, CV_32F, stream);

        match_CCORR_32F_.match(imagef_, templf_, _result, stream);
        GpuMat result = _result.getGpuMat();

        if (image.channels() == 1)
        {
            image_sums_.resize(1);
            cuda::integral(image, image_sums_[0], stream);

            image_sqsums_.resize(1);
            cuda::sqrIntegral(image, image_sqsums_[0], stream);

            int templ_sum = (int) cuda::sum(templ)[0];
            double templ_sqsum = cuda::sqrSum(templ)[0];

            matchTemplatePrepared_CCOFF_NORMED_8U(
                    templ.cols, templ.rows, image_sums_[0], image_sqsums_[0],
                    templ_sum, templ_sqsum, result, StreamAccessor::getStream(stream));
        }
        else
        {
            cuda::split(image, images_);

            image_sums_.resize(images_.size());
            image_sqsums_.resize(images_.size());
            for (int i = 0; i < image.channels(); ++i)
            {
                cuda::integral(images_[i], image_sums_[i], stream);
                cuda::sqrIntegral(images_[i], image_sqsums_[i], stream);
            }

            Scalar templ_sum = cuda::sum(templ);
            Scalar templ_sqsum = cuda::sqrSum(templ);

            switch (image.channels())
            {
            case 2:
                matchTemplatePrepared_CCOFF_NORMED_8UC2(
                        templ.cols, templ.rows,
                        image_sums_[0], image_sqsums_[0],
                        image_sums_[1], image_sqsums_[1],
                        (int)templ_sum[0], templ_sqsum[0],
                        (int)templ_sum[1], templ_sqsum[1],
                        result, StreamAccessor::getStream(stream));
                break;
            case 3:
                matchTemplatePrepared_CCOFF_NORMED_8UC3(
                        templ.cols, templ.rows,
                        image_sums_[0], image_sqsums_[0],
                        image_sums_[1], image_sqsums_[1],
                        image_sums_[2], image_sqsums_[2],
                        (int)templ_sum[0], templ_sqsum[0],
                        (int)templ_sum[1], templ_sqsum[1],
                        (int)templ_sum[2], templ_sqsum[2],
                        result, StreamAccessor::getStream(stream));
                break;
            case 4:
                matchTemplatePrepared_CCOFF_NORMED_8UC4(
                        templ.cols, templ.rows,
                        image_sums_[0], image_sqsums_[0],
                        image_sums_[1], image_sqsums_[1],
                        image_sums_[2], image_sqsums_[2],
                        image_sums_[3], image_sqsums_[3],
                        (int)templ_sum[0], templ_sqsum[0],
                        (int)templ_sum[1], templ_sqsum[1],
                        (int)templ_sum[2], templ_sqsum[2],
                        (int)templ_sum[3], templ_sqsum[3],
                        result, StreamAccessor::getStream(stream));
                break;
            default:
                CV_Error(Error::StsBadArg, "unsupported number of channels");
            }
        }
    }

    static inline Scalar scalarDiv(const Scalar& a, const Scalar& b)
    {
        Scalar s;
        for (int i = 0; i < 4; i++)
            s[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
        return s;
    }

    static void crossCorrSumChannels(
            const GpuMat& src, const GpuMat& templ, GpuMat& dst, Stream& stream,
            Ptr<Convolution>& conv, GpuMat& work,
            std::vector<GpuMat>& srcPlanes, std::vector<GpuMat>& templPlanes)
    {
        using namespace cv::cuda::device::match_template;

        const int cn = src.channels();
        CV_Assert(templ.size() == src.size() && templ.channels() == cn && src.depth() == CV_32F);
        const Size outSize(src.cols - templ.cols + 1, src.rows - templ.rows + 1);
        if (dst.size() != outSize || dst.type() != CV_32FC1)
            dst.create(outSize, CV_32FC1);

        const int thr = getTemplateThreshold(TM_CCORR, CV_32F);
        if ((int)templ.size().area() < thr)
        {
            matchTemplateNaive_CCORR_32F(src, templ, dst, cn, StreamAccessor::getStream(stream));
            return;
        }
        if (cn == 1)
        {
            conv->convolve(src, templ, dst, true, stream);
            return;
        }
        cuda::split(src, srcPlanes, stream);
        cuda::split(templ, templPlanes, stream);
        conv->convolve(srcPlanes[0], templPlanes[0], dst, true, stream);
        for (int k = 1; k < cn; ++k)
        {
            conv->convolve(srcPlanes[k], templPlanes[k], work, true, stream);
            cuda::add(dst, work, dst, noArray(), -1, stream);
        }
    }

    static void crossCorrMultichannel(
            const GpuMat& src, const GpuMat& templ, GpuMat& dst, Stream& stream,
            Ptr<Convolution>& conv,
            std::vector<GpuMat>& srcPlanes, std::vector<GpuMat>& templPlanes, std::vector<GpuMat>& dstPlanes)
    {
        using namespace cv::cuda::device::match_template;

        const int cn = src.channels();
        CV_Assert(templ.size() == src.size() && templ.channels() == cn && src.depth() == CV_32F);
        const Size outSize(src.cols - templ.cols + 1, src.rows - templ.rows + 1);
        dst.create(outSize, CV_MAKETYPE(CV_32F, cn));

        const int thr = getTemplateThreshold(TM_CCORR, CV_32F);
        cuda::split(src, srcPlanes, stream);
        cuda::split(templ, templPlanes, stream);
        dstPlanes.resize(cn);
        for (int k = 0; k < cn; ++k)
        {
            dstPlanes[k].create(outSize, CV_32FC1);
            if ((int)templ.size().area() < thr)
                matchTemplateNaive_CCORR_32F(srcPlanes[k], templPlanes[k], dstPlanes[k], 1, StreamAccessor::getStream(stream));
            else
                conv->convolve(srcPlanes[k], templPlanes[k], dstPlanes[k], true, stream);
        }
        cuda::merge(dstPlanes, dst, stream);
    }

    static void reduceSumChannelsToSingle(const GpuMat& srcMc, GpuMat& dst1c, const Size& corrSize, Stream& stream)
    {
        GpuMat flat = srcMc.reshape(1, (int)(corrSize.area()));
        GpuMat reduced;
        cuda::reduce(flat, reduced, 1, REDUCE_SUM, CV_32F, stream);
        reduced.reshape(1, corrSize.height).copyTo(dst1c, stream);
    }

    static void matchTemplateMaskSqDiffImpl(MatchTemplateMaskCtx& ctx, bool normed)
    {
        auto& b = ctx.buf;

        cuda::multiply(ctx.img_f, ctx.img_f, b.img2, 1.0, -1, ctx.stream);
        cuda::multiply(ctx.mask_f, ctx.mask_f, b.mask2, 1.0, -1, ctx.stream);
        cuda::multiply(ctx.tpl_f, ctx.mask_f, b.tpl_masked, 1.0, -1, ctx.stream);
        ctx.stream.waitForCompletion();
        const double templ2_mask2_sum = cuda::sqrSum(b.tpl_masked.reshape(1))[0];

        crossCorrSumChannels(b.img2, b.mask2, b.temp_result, ctx.stream, ctx.conv, b.work, b.spl, b.tplp);
        cuda::multiply(ctx.tpl_f, b.mask2, b.tpl_masked, 1.0, -1, ctx.stream);
        crossCorrSumChannels(ctx.img_f, b.tpl_masked, ctx.result, ctx.stream, ctx.conv, b.work, b.spl, b.tplp);

        cuda::multiplyWithScalar(ctx.result, Scalar(-2.0), ctx.result, 1.0, -1, ctx.stream);
        cuda::add(ctx.result, b.temp_result, ctx.result, noArray(), -1, ctx.stream);
        cuda::addWithScalar(ctx.result, Scalar(templ2_mask2_sum), ctx.result, noArray(), -1, ctx.stream);

        if (normed)
        {
            cuda::multiplyWithScalar(b.temp_result, Scalar(templ2_mask2_sum), b.temp_result, 1.0, -1, ctx.stream);
            cuda::sqrt(b.temp_result, b.temp_result, ctx.stream);
            cuda::divide(ctx.result, b.temp_result, ctx.result, 1.0, -1, ctx.stream);
        }
    }

    static void matchTemplateMaskCCorrImpl(MatchTemplateMaskCtx& ctx, bool normed)
    {
        auto& b = ctx.buf;

        cuda::multiply(ctx.mask_f, ctx.mask_f, b.mask2, 1.0, -1, ctx.stream);
        cuda::multiply(ctx.tpl_f, b.mask2, b.tpl_masked, 1.0, -1, ctx.stream);
        crossCorrSumChannels(ctx.img_f, b.tpl_masked, ctx.result, ctx.stream, ctx.conv, b.work, b.spl, b.tplp);

        if (!normed)
            return;

        cuda::multiply(ctx.tpl_f, ctx.mask_f, b.tpl_masked, 1.0, -1, ctx.stream);
        ctx.stream.waitForCompletion();
        const double templ2_mask2_sum = cuda::sqrSum(b.tpl_masked.reshape(1))[0];
        cuda::multiply(ctx.img_f, ctx.img_f, b.img2, 1.0, -1, ctx.stream);
        crossCorrSumChannels(b.img2, b.mask2, b.temp_result, ctx.stream, ctx.conv, b.work, b.spl, b.tplp);
        cuda::multiplyWithScalar(b.temp_result, Scalar(templ2_mask2_sum), b.temp_result, 1.0, -1, ctx.stream);
        cuda::sqrt(b.temp_result, b.temp_result, ctx.stream);
        cuda::divide(ctx.result, b.temp_result, ctx.result, 1.0, -1, ctx.stream);
    }

    static void matchTemplateMaskCCoeffNormedPart(MatchTemplateMaskCtx& ctx, const Scalar& mask_sum)
    {
        auto& b = ctx.buf;

        cuda::multiply(ctx.mask_f, b.diff_templ, b.temp_for_norm, 1.0, -1, ctx.stream);
        ctx.stream.waitForCompletion();
        const double norm_templx = cuda::norm(b.temp_for_norm, NORM_L2);

        cuda::multiply(ctx.img_f, ctx.img_f, b.img2, 1.0, -1, ctx.stream);
        cuda::multiply(ctx.mask_f, ctx.mask_f, b.mask2, 1.0, -1, ctx.stream);
        ctx.stream.waitForCompletion();
        const Scalar mask2_sum = cuda::sum(b.mask2);

        crossCorrSumChannels(b.img2, b.mask2, b.norm_imgx, ctx.stream, ctx.conv, b.work, b.spl, b.tplp);
        crossCorrMultichannel(ctx.img_f, b.mask2, b.img_mask2_corr, ctx.stream, ctx.conv, b.spl, b.tplp, b.dstp);

        const Scalar div1 = scalarDiv(Scalar::all(1.0), mask_sum);
        const Scalar div2 = scalarDiv(mask2_sum, mask_sum);
        cuda::multiplyWithScalar(b.img_mask_corr, div1, b.temp_res1, 1.0, -1, ctx.stream);
        cuda::multiplyWithScalar(b.img_mask_corr, div2, b.temp_res2, 1.0, -1, ctx.stream);
        cuda::multiplyWithScalar(b.img_mask2_corr, Scalar(2.0), b.two_m, 1.0, -1, ctx.stream);
        cuda::subtract(b.temp_res2, b.two_m, b.temp_mid, noArray(), -1, ctx.stream);
        cuda::multiply(b.temp_res1, b.temp_mid, b.temp_res, 1.0, -1, ctx.stream);

        if (ctx.cn == 1)
            cuda::add(b.norm_imgx, b.temp_res, b.norm_imgx, noArray(), -1, ctx.stream);
        else
        {
            GpuMat reduced;
            reduceSumChannelsToSingle(b.temp_res, reduced, ctx.corrSize, ctx.stream);
            cuda::add(b.norm_imgx, reduced, b.norm_imgx, noArray(), -1, ctx.stream);
        }

        cuda::sqrt(b.norm_imgx, b.norm_imgx, ctx.stream);
        cuda::multiplyWithScalar(b.norm_imgx, Scalar(norm_templx), b.norm_imgx, 1.0, -1, ctx.stream);
        cuda::divide(ctx.result, b.norm_imgx, ctx.result, 1.0, -1, ctx.stream);
    }

    static void matchTemplateMaskCCoeffImpl(MatchTemplateMaskCtx& ctx, bool normed)
    {
        auto& b = ctx.buf;

        cuda::multiply(ctx.mask_f, ctx.tpl_f, b.temp_x, 1.0, -1, ctx.stream);
        ctx.stream.waitForCompletion();
        const Scalar mask_sum = cuda::sum(ctx.mask_f);
        const Scalar mt = cuda::sum(b.temp_x);
        const Scalar ratio = scalarDiv(mt, mask_sum);

        cuda::subtractWithScalar(ctx.tpl_f, ratio, b.diff_templ, noArray(), -1, ctx.stream);
        cuda::multiply(ctx.mask_f, b.diff_templ, b.temp_x, 1.0, -1, ctx.stream);
        cuda::multiply(ctx.mask_f, b.temp_x, b.templx_mask, 1.0, -1, ctx.stream);

        crossCorrSumChannels(ctx.img_f, b.templx_mask, ctx.result, ctx.stream, ctx.conv, b.work, b.spl, b.tplp);
        crossCorrMultichannel(ctx.img_f, ctx.mask_f, b.img_mask_corr, ctx.stream, ctx.conv, b.spl, b.tplp, b.dstp);

        const Scalar st_x = cuda::sum(b.templx_mask);
        const Scalar ratio_corr = scalarDiv(st_x, mask_sum);
        cuda::multiplyWithScalar(b.img_mask_corr, ratio_corr, b.temp_res, 1.0, -1, ctx.stream);

        if (ctx.cn == 1)
            cuda::subtract(ctx.result, b.temp_res, ctx.result, noArray(), -1, ctx.stream);
        else
        {
            GpuMat reduced;
            reduceSumChannelsToSingle(b.temp_res, reduced, ctx.corrSize, ctx.stream);
            cuda::subtract(ctx.result, reduced, ctx.result, noArray(), -1, ctx.stream);
        }

        if (normed)
            matchTemplateMaskCCoeffNormedPart(ctx, mask_sum);
    }

    static void matchTemplateReadImg(InputArray _image, InputArray _templ, GpuMat& img_f, GpuMat& tpl_f, Stream& stream)
    {
        GpuMat image = _image.getGpuMat();
        GpuMat templ = _templ.getGpuMat();
        CV_Assert(image.cols >= templ.cols && image.rows >= templ.rows);
        CV_Assert(image.depth() == CV_8U || image.depth() == CV_32F);
        CV_Assert(image.type() == templ.type());

        if (image.depth() == CV_8U)
            image.convertTo(img_f, CV_32F, stream);
        else
            img_f = image;
        if (templ.depth() == CV_8U)
            templ.convertTo(tpl_f, CV_32F, stream);
        else
            tpl_f = templ;
    }

    static void matchTemplateReadMask(InputArray _mask, const GpuMat& tpl_f, GpuMat& mask_f, Stream& stream)
    {
        CV_Assert(_mask.depth() == CV_8U || _mask.depth() == CV_32F);
        GpuMat m_in = _mask.getGpuMat();
        CV_Assert(m_in.channels() == tpl_f.channels() || m_in.channels() == 1);
        CV_Assert(tpl_f.size() == m_in.size());

        if (m_in.depth() == CV_8U)
        {
            GpuMat mask_bin;
            cuda::threshold(m_in, mask_bin, 0, 1.0, THRESH_BINARY, stream);
            mask_bin.convertTo(mask_f, CV_32F, stream);
        }
        else
            mask_f = m_in;

        if (mask_f.channels() != tpl_f.channels())
        {
            std::vector<GpuMat> mv(tpl_f.channels(), mask_f);
            cuda::merge(mv, mask_f, stream);
        }
    }

    static void matchTemplateMaskRun(
            InputArray _image, InputArray _templ, OutputArray _result, int method, InputArray _mask, Stream& _stream)
    {
        Stream& stream = _stream;

        GpuMat img_f, tpl_f, mask_f;
        matchTemplateReadImg(_image, _templ, img_f, tpl_f, stream);
        matchTemplateReadMask(_mask, tpl_f, mask_f, stream);

        const Size corrSize(img_f.cols - tpl_f.cols + 1, img_f.rows - tpl_f.rows + 1);
        _result.create(corrSize, CV_32F);
        GpuMat result = _result.getGpuMat();

        Ptr<Convolution> conv = cuda::createConvolution(Size());
        MatchTemplateMaskGpuBuffers buf;
        const int cn = img_f.channels();
        MatchTemplateMaskCtx ctx{ img_f, tpl_f, mask_f, result, corrSize, cn, stream, conv, buf };

        switch (method)
        {
        case TM_SQDIFF:
            matchTemplateMaskSqDiffImpl(ctx, false);
            break;
        case TM_SQDIFF_NORMED:
            matchTemplateMaskSqDiffImpl(ctx, true);
            break;
        case TM_CCORR:
            matchTemplateMaskCCorrImpl(ctx, false);
            break;
        case TM_CCORR_NORMED:
            matchTemplateMaskCCorrImpl(ctx, true);
            break;
        case TM_CCOEFF:
            matchTemplateMaskCCoeffImpl(ctx, false);
            break;
        case TM_CCOEFF_NORMED:
            matchTemplateMaskCCoeffImpl(ctx, true);
            break;
        default:
            CV_Error(Error::StsBadFlag, "Unsupported match template method for masked match");
        }
    }
}

void cv::cuda::matchTemplate(InputArray image, InputArray templ, OutputArray result, int method, InputArray mask, Stream& stream)
{
    matchTemplateMaskRun(image, templ, result, method, mask, stream);
}

Ptr<cuda::TemplateMatching> cv::cuda::createTemplateMatching(int srcType, int method, Size user_block_size)
{
    const int sdepth = CV_MAT_DEPTH(srcType);

    CV_Assert( sdepth == CV_8U || sdepth == CV_32F );

    if (sdepth == CV_32F)
    {
        switch (method)
        {
        case TM_SQDIFF:
            return makePtr<Match_SQDIFF_32F>();

        case TM_CCORR:
            return makePtr<Match_CCORR_32F>(user_block_size);

        default:
            CV_Error( Error::StsBadFlag, "Unsopported method" );
            return Ptr<cuda::TemplateMatching>();
        }
    }
    else
    {
        switch (method)
        {
        case TM_SQDIFF:
            return makePtr<Match_SQDIFF_8U>(user_block_size);

        case TM_SQDIFF_NORMED:
            return makePtr<Match_SQDIFF_NORMED_8U>(user_block_size);

        case TM_CCORR:
            return makePtr<Match_CCORR_8U>(user_block_size);

        case TM_CCORR_NORMED:
            return makePtr<Match_CCORR_NORMED_8U>(user_block_size);

        case TM_CCOEFF:
            return makePtr<Match_CCOEFF_8U>(user_block_size);

        case TM_CCOEFF_NORMED:
            return makePtr<Match_CCOEFF_NORMED_8U>(user_block_size);

        default:
            CV_Error( Error::StsBadFlag, "Unsopported method" );
            return Ptr<cuda::TemplateMatching>();
        }
    }
}

#endif
