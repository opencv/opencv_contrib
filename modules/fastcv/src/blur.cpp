/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class FcvGaussianBlurLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvGaussianBlurLoop_Invoker(const uchar* _src_data, size_t _src_step, uchar* _dst_data, size_t _dst_step, int _width,
        int _height, int _ksize, int _depth, fcvBorderType _fcvBorder, int _fcvBorderValue) :
        cv::ParallelLoopBody(), src_data(_src_data), src_step(_src_step), dst_data(_dst_data), dst_step(_dst_step), width(_width),
        height(_height), ksize(_ksize), depth(_depth), fcvBorder(_fcvBorder), fcvBorderValue(_fcvBorderValue)
    {
        half_ksize = ksize/2;
        fcvFuncType = FCV_MAKETYPE(ksize,depth);
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int topLines    = 0;
        int rangeHeight = range.end-range.start;

        if(range.start >= half_ksize)
        {
            topLines  += half_ksize;
            rangeHeight += half_ksize;
        }

        if(range.end <= height-half_ksize)
        {
            rangeHeight += half_ksize;
        }

        const uchar* src = src_data + (range.start-topLines)*src_step;
        std::vector<uchar> dst(dst_step*rangeHeight);

        if (fcvFuncType == FCV_MAKETYPE(3,CV_8U))
            fcvFilterGaussian3x3u8_v4(src, width, rangeHeight, src_step, dst.data(), dst_step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_8U))
            fcvFilterGaussian5x5u8_v3(src, width, rangeHeight, src_step, dst.data(), dst_step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_16S))
            fcvFilterGaussian5x5s16_v3((int16_t*)src, width, rangeHeight, src_step, (int16_t*)dst.data(), dst_step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_32S))
            fcvFilterGaussian5x5s32_v3((int32_t*)src, width, rangeHeight, src_step, (int32_t*)dst.data(), dst_step, fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(11,CV_8U))
            fcvFilterGaussian11x11u8_v2(src, width, rangeHeight, src_step, dst.data(), dst_step, fcvBorder);

        uchar *dptr = dst_data + range.start * dst_step;
        uchar *sptr = dst.data() + topLines * dst_step;
        memcpy(dptr, sptr, (range.end - range.start) * dst_step);
    }

    private:
    const uchar*    src_data;
    const size_t    src_step;
    uchar*          dst_data;
    const size_t    dst_step;
    const int       width;
    const int       height;
    const int       ksize;
    const int       depth;
    int             half_ksize;
    int             fcvFuncType;
    fcvBorderType   fcvBorder;
    int             fcvBorderValue;

    FcvGaussianBlurLoop_Invoker(const FcvGaussianBlurLoop_Invoker &);  // = delete;
    const FcvGaussianBlurLoop_Invoker& operator= (const FcvGaussianBlurLoop_Invoker &);  // = delete;
};

void gaussianBlur(cv::InputArray _src, cv::OutputArray _dst, int kernel_size, bool blur_border)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && CV_MAT_CN(_src.type()) == 1);

    Size size = _src.size();
    int type  = _src.type();
    _dst.create( size, type );

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    int nStripes = src.rows / 80 == 0 ? 1 : src.rows / 80;

    fcvBorderType fcvBorder = blur_border ? FASTCV_BORDER_ZERO_PADDING : FASTCV_BORDER_UNDEFINED;

    if (((type == CV_8UC1)  && ((kernel_size == 3) || (kernel_size == 5) || (kernel_size == 11)))  ||
        ((type == CV_16SC1) && (kernel_size == 5)) ||
        ((type == CV_32SC1) && (kernel_size == 5)))
    {
        cv::parallel_for_(cv::Range(0, src.rows),
            FcvGaussianBlurLoop_Invoker(src.data, src.step, dst.data, dst.step, src.cols, src.rows, kernel_size,
            src.depth(), fcvBorder, 0), nStripes);
    }
    else
        CV_Error(cv::Error::StsBadArg, cv::format("Src type %d, kernel size %d is not supported", type, kernel_size));
}

class FcvFilter2DLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvFilter2DLoop_Invoker(const uchar* _src_data, size_t _src_step, uchar* _dst_data, size_t _dst_step, const int _ddepth,
        int _width, int _height, uchar* _kernel,int _ksize ) :
        cv::ParallelLoopBody(), src_data(_src_data), src_step(_src_step), dst_data(_dst_data), dst_step(_dst_step),
        ddepth(_ddepth), width(_width),height(_height), kernel(_kernel), ksize(_ksize)
    {
        half_ksize = ksize/2;
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int topLines    = 0;
        int rangeHeight = range.end-range.start;

        if(range.start >= half_ksize)
        {
            topLines  += half_ksize;
            rangeHeight += half_ksize;
        }

        if(range.end <= height-half_ksize)
        {
            rangeHeight += half_ksize;
        }

        const uchar *src = src_data + (range.start - topLines) * src_step;
        std::vector<uchar> dst(dst_step*rangeHeight);

        if (ddepth == CV_8U)
            fcvFilterCorrNxNu8((int8_t*)kernel, ksize, 0, src, width, rangeHeight, src_step, dst.data(), dst_step);
        else if (ddepth == CV_16S)
            fcvFilterCorrNxNu8s16((int8_t*)kernel, ksize, 0, src, width, rangeHeight, src_step, (int16_t*)dst.data(), dst_step);
        else if (ddepth == CV_32F)
            fcvFilterCorrNxNu8f32((float32_t*)kernel, ksize, src, width, rangeHeight, src_step, (float32_t*)dst.data(), dst_step);

        uchar *dptr = dst_data + range.start * dst_step;
        uchar *sptr = dst.data() + topLines * dst_step;
        memcpy(dptr, sptr, (range.end - range.start) * dst_step);
    }

    private:
    const uchar*    src_data;
    const size_t    src_step;
    uchar*          dst_data;
    const size_t    dst_step;
    const int       ddepth;
    const int       width;
    const int       height;
    uchar*          kernel;
    const int       ksize;
    int             half_ksize;

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

    int nStripes = src.rows / 80 == 0 ? 1 : src.rows / 80;

    switch (ddepth)
    {
        case CV_8U:
        case CV_16S:
        {
            CV_Assert(CV_MAT_DEPTH(kernel.type()) == CV_8S);

            cv::parallel_for_(cv::Range(0, src.rows),
            FcvFilter2DLoop_Invoker(src.data, src.step, dst.data, dst.step, ddepth, src.cols, src.rows, kernel.data, ksize.width),
            nStripes);
            break;
        }
        case CV_32F:
        {
            CV_Assert(CV_MAT_DEPTH(kernel.type()) == CV_32F);

            cv::parallel_for_(cv::Range(0, src.rows),
            FcvFilter2DLoop_Invoker(src.data, src.step, dst.data, dst.step, ddepth, src.cols, src.rows, kernel.data, ksize.width),
            nStripes);
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

class FcvSepFilter2DLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvSepFilter2DLoop_Invoker(const uchar* _src_data, size_t _src_step, uchar* _dst_data, size_t _dst_step, const int _ddepth,
        int _width, int _height, uchar* _kernelX, int _kernelXSize, uchar* _kernelY,int _kernelYSize) :
        cv::ParallelLoopBody(), src_data(_src_data), src_step(_src_step), dst_data(_dst_data), dst_step(_dst_step), ddepth(_ddepth),
        width(_width), height(_height), kernelX(_kernelX), kernelXSize(_kernelXSize), kernelY(_kernelY), kernelYSize(_kernelYSize)
    {
        half_ksize = kernelYSize/2;
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int topLines    = 0;
        int rangeHeight = range.end-range.start;

        if(range.start >= half_ksize)
        {
            topLines  += half_ksize;
            rangeHeight += half_ksize;
        }

        if(range.end <= height-half_ksize)
        {
            rangeHeight += half_ksize;
        }

        const uchar *src = src_data + (range.start - topLines) * src_step;
        std::vector<uchar> dst(dst_step*rangeHeight);

        switch (ddepth)
        {
            case CV_8U:
            {
                fcvFilterCorrSepMxNu8((int8_t*)kernelX, kernelXSize, (int8_t*)kernelY, kernelYSize, 0, src, width, rangeHeight,
                    src_step, dst.data(), dst_step);
                break;
            }
            case CV_16S:
            {
                std::vector<int16_t> tmpImage(width*(rangeHeight+kernelXSize-1));
                switch (kernelXSize)
                {
                    case 9:
                    {
                        fcvFilterCorrSep9x9s16_v2((int16_t*)kernelX, (int16_t*)src, width, rangeHeight, src_step,
                            tmpImage.data(), (int16_t*)dst.data(), dst_step);
                        break;
                    }
                    case 11:
                    {
                        fcvFilterCorrSep11x11s16_v2((int16_t*)kernelX, (int16_t*)src, width, rangeHeight, src_step,
                            tmpImage.data(), (int16_t*)dst.data(), dst_step);
                        break;
                    }
                    case 13:
                    {
                        fcvFilterCorrSep13x13s16_v2((int16_t*)kernelX, (int16_t*)src, width, rangeHeight, src_step,
                            tmpImage.data(), (int16_t*)dst.data(), dst_step);
                        break;
                    }
                    case 15:
                    {
                        fcvFilterCorrSep15x15s16_v2((int16_t*)kernelX, (int16_t*)src, width, rangeHeight, src_step,
                            tmpImage.data(), (int16_t*)dst.data(), dst_step);
                        break;
                    }
                    case 17:
                    {
                        fcvFilterCorrSep17x17s16_v2((int16_t*)kernelX, (int16_t*)src, width, rangeHeight, src_step,
                            tmpImage.data(), (int16_t*)dst.data(), dst_step);
                        break;
                    }

                    default:
                    {
                        fcvFilterCorrSepNxNs16((int16_t*)kernelX, kernelXSize, (int16_t*)src, width, rangeHeight, src_step,
                            tmpImage.data(), (int16_t*)dst.data(), dst_step);
                        break;
                    }
                }
                break;
            }
            default:
            {
                CV_Error(cv::Error::StsBadArg, cv::format("Dst type:%s is not supported", depthToString(ddepth)));
                break;
            }
        }

        uchar *dptr = dst_data + range.start * dst_step;
        uchar *sptr = dst.data() + topLines * dst_step;
        memcpy(dptr, sptr, (range.end - range.start) * dst_step);
    }

    private:
    const uchar*    src_data;
    const size_t    src_step;
    uchar*          dst_data;
    const size_t    dst_step;
    const int       ddepth;
    const int       width;
    const int       height;
    uchar*          kernelX;
    const int       kernelXSize;
    uchar*          kernelY;
    const int       kernelYSize;
    int             half_ksize;

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

    int nStripes = src.rows / 80 == 0 ? 1 : src.rows / 80;
    switch (ddepth)
    {
        case CV_8U:
        {
            cv::parallel_for_(cv::Range(0, src.rows),
            FcvSepFilter2DLoop_Invoker(src.data, src.step, dst.data, dst.step, ddepth, src.cols, src.rows, kernelX.data,
                kernelX.size().width, kernelY.data, kernelY.size().width),nStripes);
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

            cv::parallel_for_(cv::Range(0, src.rows),
            FcvSepFilter2DLoop_Invoker(src.data, src.step, dst.data, dst.step, ddepth, src.cols, src.rows, kernelX.data,
                kernelX.size().width, kernelY.data, kernelY.size().width),nStripes);
            break;
        }
        default:
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Dst type:%s is not supported", depthToString(ddepth)));
            break;
        }
    }
}

} // fastcv::
} // cv::