/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class FcvGaussianBlurLoop_Invoker : public ParallelLoopBody
{
    public:

    FcvGaussianBlurLoop_Invoker(const Mat& _src, Mat& _dst, int _ksize, fcvBorderType _fcvBorder, int _fcvBorderValue) :
        ParallelLoopBody(), src(_src),dst(_dst), ksize(_ksize), fcvBorder(_fcvBorder), fcvBorderValue(_fcvBorderValue)
    {
        width       = src.cols;
        height      = src.rows;
        halfKsize   = ksize / 2;
        fcvFuncType = FCV_MAKETYPE(ksize, src.depth());
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        int topLines     = 0;
        int rangeHeight  = range.end-range.start;
        int paddedHeight = rangeHeight;

        if(range.start != 0)
        {
            topLines     += halfKsize;
            paddedHeight += halfKsize;
        }

        if(range.end != height)
        {
            paddedHeight += halfKsize;
        }

        const Mat srcPadded = src(Rect(0, range.start - topLines, width, paddedHeight));
        Mat dstPadded       = Mat(paddedHeight, width, dst.depth());

        if (fcvFuncType == FCV_MAKETYPE(3,CV_8U))
            fcvFilterGaussian3x3u8_v4(srcPadded.data, width, paddedHeight, srcPadded.step, dstPadded.data, dstPadded.step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_8U))
            fcvFilterGaussian5x5u8_v3(srcPadded.data, width, paddedHeight, srcPadded.step, dstPadded.data, dstPadded.step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_16S))
            fcvFilterGaussian5x5s16_v3((int16_t*)srcPadded.data, width, paddedHeight, srcPadded.step, (int16_t*)dstPadded.data,
                dstPadded.step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_32S))
            fcvFilterGaussian5x5s32_v3((int32_t*)srcPadded.data, width, paddedHeight, srcPadded.step, (int32_t*)dstPadded.data,
                dstPadded.step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(11,CV_8U))
            fcvFilterGaussian11x11u8_v2(srcPadded.data, width, rangeHeight, srcPadded.step, dstPadded.data, dstPadded.step, fcvBorder);

        // Only copy center part back to output image and ignore the padded lines
        Mat temp1 = dstPadded(Rect(0, topLines, width, rangeHeight));
        Mat temp2 = dst(Rect(0, range.start, width, rangeHeight));
        temp1.copyTo(temp2);
    }

    private:
    const Mat&      src;
    Mat&            dst;
    int             width;
    int             height;
    const int       ksize;
    int             halfKsize;
    int             fcvFuncType;
    fcvBorderType   fcvBorder;
    int             fcvBorderValue;

    FcvGaussianBlurLoop_Invoker(const FcvGaussianBlurLoop_Invoker &);  // = delete;
    const FcvGaussianBlurLoop_Invoker& operator= (const FcvGaussianBlurLoop_Invoker &);  // = delete;
};

void gaussianBlur(InputArray _src, OutputArray _dst, int kernel_size, bool blur_border)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && CV_MAT_CN(_src.type()) == 1);

    Size size = _src.size();
    int type  = _src.type();
    _dst.create( size, type );

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    int nThreads = getNumThreads();
    int nStripes = (nThreads > 1) ? ((src.rows > 60) ? 3 * nThreads : 1) : 1;

    fcvBorderType fcvBorder = blur_border ? FASTCV_BORDER_ZERO_PADDING : FASTCV_BORDER_UNDEFINED;

    if (((type == CV_8UC1)  && ((kernel_size == 3) || (kernel_size == 5) || (kernel_size == 11)))  ||
        ((type == CV_16SC1) && (kernel_size == 5)) ||
        ((type == CV_32SC1) && (kernel_size == 5)))
    {
        parallel_for_(Range(0, src.rows), FcvGaussianBlurLoop_Invoker(src, dst, kernel_size, fcvBorder, 0), nStripes);
    }
    else
        CV_Error(cv::Error::StsBadArg, cv::format("Src type %d, kernel size %d is not supported", type, kernel_size));
}

class FcvFilter2DLoop_Invoker : public ParallelLoopBody
{
    public:

    FcvFilter2DLoop_Invoker(const Mat& _src, Mat& _dst, const Mat& _kernel) :
        ParallelLoopBody(), src(_src), dst(_dst), kernel(_kernel)
    {
        width     = src.cols;
        height    = src.rows;
        ksize     = kernel.size().width;
        halfKsize = ksize/2;
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        int topLines     = 0;
        int rangeHeight  = range.end-range.start;
        int paddedHeight = rangeHeight;

        if(range.start >= halfKsize)
        {
            topLines    += halfKsize;
            paddedHeight += halfKsize;
        }

        if(range.end <= height-halfKsize)
        {
            paddedHeight += halfKsize;
        }

        const Mat srcPadded = src(Rect(0, range.start - topLines, width, paddedHeight));
        Mat dstPadded       = Mat(paddedHeight, width, dst.depth());

        if (dst.depth() == CV_8U)
            fcvFilterCorrNxNu8((int8_t*)kernel.data, ksize, 0, srcPadded.data, width, paddedHeight, srcPadded.step,
                dstPadded.data, dstPadded.step);
        else if (dst.depth() == CV_16S)
            fcvFilterCorrNxNu8s16((int8_t*)kernel.data, ksize, 0, srcPadded.data, width, paddedHeight, srcPadded.step,
                (int16_t*)dstPadded.data, dstPadded.step);
        else if (dst.depth() == CV_32F)
            fcvFilterCorrNxNu8f32((float32_t*)kernel.data, ksize, srcPadded.data, width, paddedHeight, srcPadded.step,
                (float32_t*)dstPadded.data, dstPadded.step);

        // Only copy center part back to output image and ignore the padded lines
        Mat temp1 = dstPadded(Rect(0, topLines, width, rangeHeight));
        Mat temp2 = dst(Rect(0, range.start, width, rangeHeight));
        temp1.copyTo(temp2);
    }

    private:
    const Mat&  src;
    Mat&        dst;
    const Mat&  kernel;
    int         width;
    int         height;
    int         ksize;
    int         halfKsize;

    FcvFilter2DLoop_Invoker(const FcvFilter2DLoop_Invoker &);  // = delete;
    const FcvFilter2DLoop_Invoker& operator= (const FcvFilter2DLoop_Invoker &);  // = delete;
};

void filter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernel)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);

    Mat kernel = _kernel.getMat();
    Size ksize = kernel.size();
    CV_Assert(ksize.width == ksize.height);
    CV_Assert(ksize.width % 2 == 1);

    _dst.create(_src.size(), ddepth);
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    int nThreads = getNumThreads();
    int nStripes = (nThreads > 1) ? ((src.rows > 60) ? 3 * nThreads : 1) : 1;

    switch (ddepth)
    {
        case CV_8U:
        case CV_16S:
        {
            CV_Assert(CV_MAT_DEPTH(kernel.type()) == CV_8S);
            parallel_for_(Range(0, src.rows), FcvFilter2DLoop_Invoker(src, dst, kernel), nStripes);
            break;
        }
        case CV_32F:
        {
            CV_Assert(CV_MAT_DEPTH(kernel.type()) == CV_32F);
            parallel_for_(Range(0, src.rows), FcvFilter2DLoop_Invoker(src, dst, kernel), nStripes);
            break;
        }
        default:
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Kernel Size:%d, Dst type:%s is not supported", ksize.width,
                depthToString(ddepth)));
            break;
        }
    }
}

class FcvSepFilter2DLoop_Invoker : public ParallelLoopBody
{
    public:

    FcvSepFilter2DLoop_Invoker(const Mat& _src, Mat& _dst, const Mat& _kernelX, const Mat& _kernelY) :
        ParallelLoopBody(), src(_src), dst(_dst), kernelX(_kernelX), kernelY(_kernelY)
    {
        width       = src.cols;
        height      = src.rows;
        kernelXSize = kernelX.size().width;
        kernelYSize = kernelY.size().width;
        halfKsize   = kernelXSize/2;
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        int topLines     = 0;
        int rangeHeight  = range.end-range.start;
        int paddedHeight = rangeHeight;

        if(range.start >= halfKsize)
        {
            topLines     += halfKsize;
            paddedHeight += halfKsize;
        }

        if(range.end <= height-halfKsize)
        {
            paddedHeight += halfKsize;
        }

        const Mat srcPadded = src(Rect(0, range.start - topLines, width, paddedHeight));
        Mat dstPadded       = Mat(paddedHeight, width, dst.depth());

        switch (dst.depth())
        {
            case CV_8U:
            {
                fcvFilterCorrSepMxNu8((int8_t*)kernelX.data, kernelXSize, (int8_t*)kernelY.data, kernelYSize, 0, srcPadded.data,
                    width, paddedHeight, srcPadded.step, dstPadded.data, dstPadded.step);
                break;
            }
            case CV_16S:
            {
                std::vector<int16_t> tmpImage(width * (paddedHeight + kernelXSize - 1));
                switch (kernelXSize)
                {
                    case 9:
                    {
                        fcvFilterCorrSep9x9s16_v2((int16_t*)kernelX.data, (int16_t*)srcPadded.data, width, paddedHeight,
                            srcPadded.step, tmpImage.data(), (int16_t*)dstPadded.data, dstPadded.step);
                        break;
                    }
                    case 11:
                    {
                        fcvFilterCorrSep11x11s16_v2((int16_t*)kernelX.data, (int16_t*)srcPadded.data, width, paddedHeight,
                            srcPadded.step, tmpImage.data(), (int16_t*)dstPadded.data, dstPadded.step);
                        break;
                    }
                    case 13:
                    {
                        fcvFilterCorrSep13x13s16_v2((int16_t*)kernelX.data, (int16_t*)srcPadded.data, width, paddedHeight,
                            srcPadded.step, tmpImage.data(), (int16_t*)dstPadded.data, dstPadded.step);
                        break;
                    }
                    case 15:
                    {
                        fcvFilterCorrSep15x15s16_v2((int16_t*)kernelX.data, (int16_t*)srcPadded.data, width, paddedHeight,
                            srcPadded.step, tmpImage.data(), (int16_t*)dstPadded.data, dstPadded.step);
                        break;
                    }
                    case 17:
                    {
                        fcvFilterCorrSep17x17s16_v2((int16_t*)kernelX.data, (int16_t*)srcPadded.data, width, paddedHeight,
                            srcPadded.step, tmpImage.data(), (int16_t*)dstPadded.data, dstPadded.step);
                        break;
                    }

                    default:
                    {
                        fcvFilterCorrSepNxNs16((int16_t*)kernelX.data, kernelXSize, (int16_t*)srcPadded.data, width, paddedHeight,
                            srcPadded.step, tmpImage.data(), (int16_t*)dstPadded.data, dstPadded.step);
                        break;
                    }
                }
                break;
            }
            default:
            {
                CV_Error(cv::Error::StsBadArg, cv::format("Dst type:%s is not supported", depthToString(dst.depth())));
                break;
            }
        }

        // Only copy center part back to output image and ignore the padded lines
        Mat temp1 = dstPadded(Rect(0, topLines, width, rangeHeight));
        Mat temp2 = dst(Rect(0, range.start, width, rangeHeight));
        temp1.copyTo(temp2);
    }

    private:
    const Mat&  src;
    Mat&        dst;
    int         width;
    int         height;
    const Mat&  kernelX;
    const Mat&  kernelY;
    int         kernelXSize;
    int         kernelYSize;
    int         halfKsize;

    FcvSepFilter2DLoop_Invoker(const FcvSepFilter2DLoop_Invoker &);  // = delete;
    const FcvSepFilter2DLoop_Invoker& operator= (const FcvSepFilter2DLoop_Invoker &);  // = delete;
};

void sepFilter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernelX, InputArray _kernelY)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src.empty() && (_src.type() == CV_8UC1 || _src.type() == CV_16SC1));
    _dst.create(_src.size(), ddepth);
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    Mat kernelX = _kernelX.getMat();
    Mat kernelY = _kernelY.getMat();

    int nThreads = getNumThreads();
    int nStripes = (nThreads > 1) ? ((src.rows > 60) ? 3 * nThreads : 1) : 1;

    switch (ddepth)
    {
        case CV_8U:
        {
            cv::parallel_for_(cv::Range(0, src.rows), FcvSepFilter2DLoop_Invoker(src, dst, kernelX, kernelY), nStripes);
            break;
        }
        case CV_16S:
        {
            CV_Assert(CV_MAT_DEPTH(src.type()) == CV_16S);
            CV_Assert(kernelX.size() == kernelY.size());
            // kernalX and kernelY shhould be same.
            Mat diff;
            absdiff(kernelX, kernelY, diff);
            CV_Assert(countNonZero(diff) == 0);

            cv::parallel_for_(cv::Range(0, src.rows), FcvSepFilter2DLoop_Invoker(src, dst, kernelX, kernelY), nStripes);
            break;
        }
        default:
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Dst type:%s is not supported", depthToString(ddepth)));
            break;
        }
    }
}

void normalizeLocalBox(InputArray _src, OutputArray _dst, Size pSize, bool useStdDev)
{
    CV_Assert(!_src.empty());
    int type = _src.type();
    CV_Assert(type == CV_8UC1 || type == CV_32FC1);

    Size size = _src.size();
    int dst_type = type == CV_8UC1 ? CV_8SC1 : CV_32FC1;
    _dst.create(size, dst_type);

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    if(type == CV_8UC1)
        fcvNormalizeLocalBoxu8(src.data, src.cols, src.rows, src.step[0],
                              pSize.width, pSize.height, useStdDev, (int8_t*)dst.data, dst.step[0]);
    else if(type == CV_32FC1)
        fcvNormalizeLocalBoxf32((float*)src.data, src.cols, src.rows, src.step[0],
                              pSize.width, pSize.height, useStdDev, (float*)dst.data, dst.step[0]);
}

} // fastcv::
} // cv::