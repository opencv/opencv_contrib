/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class RemapParallel : public cv::ParallelLoopBody {
public:
    RemapParallel(int src_type, const uint8_t* src, uint32_t srcWidth, uint32_t srcHeight, uint32_t srcStride, uint8_t* dst,
                uint32_t dstWidth, uint32_t dstHeight, uint32_t dstStride, const float32_t* __restrict  mapX,
                const float32_t* __restrict mapY, uint32_t mapStride, fcvInterpolationType interpolation, uint8_t borderValue)
                : src_type_(src_type), src_(src), srcWidth_(srcWidth), srcHeight_(srcHeight), srcStride_(srcStride), dst_(dst), dstWidth_(dstWidth),
                dstHeight_(dstHeight), dstStride_(dstStride), mapX_(mapX), mapY_(mapY), mapStride_(mapStride),
                fcvInterpolation_(interpolation), borderValue_(borderValue) {}

    void operator()(const cv::Range& range) const override {
        CV_UNUSED(srcHeight_);
        CV_UNUSED(dstHeight_);
        int rangeHeight = range.end-range.start;
        fcvStatus   status = FASTCV_SUCCESS;
        if(src_type_==CV_8UC1)
        {
            status = fcvRemapu8_v2(src_ + range.start*srcStride_, srcWidth_, rangeHeight, srcStride_, dst_ + range.start*dstStride_,
                            srcWidth_, rangeHeight, dstStride_, mapX_, mapY_, mapStride_, fcvInterpolation_, FASTCV_BORDER_CONSTANT, borderValue_);
        }
        else if(src_type_==CV_8UC4)
        {
            if(fcvInterpolation_ == FASTCV_INTERPOLATION_TYPE_BILINEAR)
            {
                fcvRemapRGBA8888BLu8(src_ + range.start*srcStride_, srcWidth_, rangeHeight, srcStride_, dst_ + range.start*dstStride_, dstWidth_, rangeHeight,
                                    dstStride_, mapX_, mapY_, mapStride_);
            }
            else if(fcvInterpolation_ == FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR)
            {
                fcvRemapRGBA8888NNu8(src_ + range.start*srcStride_, srcWidth_, rangeHeight, srcStride_, dst_ + range.start*dstStride_, dstWidth_, rangeHeight,
                                    dstStride_, mapX_, mapY_, mapStride_);
            }
        }

        if(status!=FASTCV_SUCCESS)
        {
            std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
            CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
        }
    }

private:
    int src_type_;
    const uint8_t* src_;
    uint32_t srcWidth_;
    uint32_t srcHeight_;
    uint32_t srcStride_;
    uint8_t* dst_;
    uint32_t dstWidth_;
    uint32_t dstHeight_;
    uint32_t dstStride_;
    const float32_t* __restrict mapX_;
    const float32_t* __restrict mapY_;
    uint32_t mapStride_;
    fcvInterpolationType fcvInterpolation_;
    uint8_t borderValue_;
};

void remap(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _map1, cv::InputArray _map2,
                      int interpolation, int borderValue)
{
    INITIALIZATION_CHECK;

    CV_Assert(_src.type() == CV_8UC1);
    CV_Assert(_map1.type()==CV_32FC1);
    CV_Assert(interpolation == cv::InterpolationFlags::INTER_NEAREST || interpolation == cv::InterpolationFlags::INTER_LINEAR);
    CV_Assert(!_map1.empty() && !_map2.empty());
    CV_Assert(_map1.size() == _map2.size());
    CV_Assert(borderValue >= 0 && borderValue < 256);

    Size size = _map1.size();
    int type = _src.type();
    _dst.create( size, type);

    Mat src = _src.getMat();
    Mat map1 = _map1.getMat();
    Mat map2 = _map2.getMat();
    Mat dst = _dst.getMat();
    CV_Assert(map1.step == map2.step);
    fcvStatus               status = FASTCV_SUCCESS;
    fcvInterpolationType    fcvInterpolation;

    if(interpolation==cv::InterpolationFlags::INTER_NEAREST)
        fcvInterpolation = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
    else
        fcvInterpolation = FASTCV_INTERPOLATION_TYPE_BILINEAR;


    cv::parallel_for_(cv::Range(0, src.rows), RemapParallel(CV_8UC1, src.data, src.cols, src.rows, src.step, dst.data, dst.cols, dst.rows, dst.step,
    (float32_t*)map1.data, (float32_t*)map2.data, map1.step, fcvInterpolation, borderValue), (src.cols*src.rows)/(double)(1 << 16));

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

void remapRGBA(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _map1, cv::InputArray _map2, int interpolation)
{
    INITIALIZATION_CHECK;

    CV_Assert(_src.type() == CV_8UC4);
    CV_Assert(_map1.type()==CV_32FC1);
    CV_Assert(interpolation == cv::InterpolationFlags::INTER_NEAREST || interpolation == cv::InterpolationFlags::INTER_LINEAR);
    CV_Assert(!_map1.empty() && !_map2.empty());
    CV_Assert(_map1.size() == _map2.size());

    Size size = _map1.size();
    int type = _src.type();
    _dst.create( size, type);

    Mat src = _src.getMat();
    Mat map1 = _map1.getMat();
    Mat map2 = _map2.getMat();
    Mat dst = _dst.getMat();
    CV_Assert(map1.step == map2.step);
    fcvStatus               status = FASTCV_SUCCESS;
    fcvInterpolationType    fcvInterpolation;

    if(interpolation==cv::InterpolationFlags::INTER_NEAREST)
        fcvInterpolation = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
    else
        fcvInterpolation = FASTCV_INTERPOLATION_TYPE_BILINEAR;

    cv::parallel_for_(cv::Range(0, src.rows), RemapParallel(CV_8UC4, src.data, src.cols, src.rows, src.step, dst.data, dst.cols, dst.rows, dst.step,
    (float32_t*)map1.data, (float32_t*)map2.data, map1.step, fcvInterpolation, 0), (src.cols*src.rows)/(double)(1 << 16) );

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::
