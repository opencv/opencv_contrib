/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void sobelPyramid(InputArrayOfArrays _pyr, OutputArrayOfArrays _dx, OutputArrayOfArrays _dy, int outType)
{
    INITIALIZATION_CHECK;

    CV_Assert(_pyr.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
              _pyr.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
              _pyr.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);
    CV_Assert(_dx.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
              _dx.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
              _dx.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);
    CV_Assert(_dy.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
              _dy.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
              _dy.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);

    std::vector<cv::Mat> pyr;
    _pyr.getMatVector(pyr);
    size_t nLevels = pyr.size();

    CV_Assert(!pyr.empty());

    // this should be smaller I guess
    CV_Assert(nLevels > 0 && nLevels < 16);

    for (size_t i = 0; i < nLevels; i++)
    {
        // fcvPyramidLeved does not support other cases
        CV_Assert(pyr[i].isContinuous());
        CV_Assert(pyr[i].type() == CV_8UC1);
    }

    CV_Assert(outType == CV_8S || outType == CV_16S || outType == CV_32F);

    std::vector<fcvPyramidLevel> lpyr;
    for (size_t i = 0; i < nLevels; i++)
    {
        fcvPyramidLevel lev;
        lev.width  = pyr[i].cols;
        lev.height = pyr[i].rows;
        lev.ptr    = pyr[i].data;
        lpyr.push_back(lev);
    }

    std::vector<fcvPyramidLevel> ldx(nLevels), ldy(nLevels);
    int pyrElemSz = (outType == CV_8S ) ? 1 :
                    (outType == CV_16S) ? 2 :
                    (outType == CV_32F) ? 4 : 0;
    int retCodex = fcvPyramidAllocate(ldx.data(), pyr[0].cols, pyr[0].rows, pyrElemSz, nLevels, 1);
    if (retCodex != 0)
    {
        CV_Error(cv::Error::StsInternal, cv::format("fcvPyramidAllocate returned code %d", retCodex));
    }
    int retCodey = fcvPyramidAllocate(ldy.data(), pyr[0].cols, pyr[0].rows, pyrElemSz, nLevels, 1);
    if (retCodey != 0)
    {
        CV_Error(cv::Error::StsInternal, cv::format("fcvPyramidAllocate returned code %d", retCodey));
    }

    int returnCode = -1;
    switch (outType)
    {
    case CV_8S:  returnCode = fcvPyramidSobelGradientCreatei8 (lpyr.data(), ldx.data(), ldy.data(), nLevels);
        break;
    case CV_16S: returnCode = fcvPyramidSobelGradientCreatei16(lpyr.data(), ldx.data(), ldy.data(), nLevels);
        break;
    case CV_32F: returnCode = fcvPyramidSobelGradientCreatef32(lpyr.data(), ldx.data(), ldy.data(), nLevels);
        break;
    default:
        break;
    }

    if (returnCode != 0)
    {
        CV_Error(cv::Error::StsInternal, cv::format("FastCV returned code %d", returnCode));
    }

    // resize arrays of Mats
    _dx.create(1, nLevels, /* type does not matter here */ -1, -1);
    _dy.create(1, nLevels, /* type does not matter here */ -1, -1);

    for (size_t i = 0; i < nLevels; i++)
    {
        cv::Mat dx((int)ldx[i].height, (int)ldx[i].width, outType, (uchar*)ldx[i].ptr);
        _dx.create(pyr[i].size(), outType, i);
        dx.copyTo(_dx.getMat(i));

        cv::Mat dy((int)ldy[i].height, (int)ldy[i].width, outType, (uchar*)ldy[i].ptr);
        _dy.create(pyr[i].size(), outType, i);
        dy.copyTo(_dy.getMat(i));
    }

    fcvPyramidDelete(ldx.data(), nLevels, 0);
    fcvPyramidDelete(ldy.data(), nLevels, 0);
}


void buildPyramid(InputArray _src, OutputArrayOfArrays _pyr, int nLevels, bool scaleBy2, int borderType, uint8_t borderValue)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && (_src.type() == CV_8UC1 || _src.type() == CV_32FC1));
    CV_Assert(_src.step() % 8 == 0);

    cv::Mat src = _src.getMat();
    bool useFloat = src.depth() == CV_32F;
    int bytesPerPixel = useFloat ? 4 : 1;

    CV_Assert(_pyr.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
              _pyr.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
              _pyr.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);

    // this should be smaller I guess
    CV_Assert(nLevels > 0 && nLevels < 16);

    if (useFloat && !scaleBy2)
    {
        CV_Error( cv::Error::StsBadArg, "ORB scale is not supported for float images (fcvPyramidCreatef32_v2)");
    }

    fcvPyramidScale scaleOption = scaleBy2 ? FASTCV_PYRAMID_SCALE_HALF : FASTCV_PYRAMID_SCALE_ORB;
    fcvBorderType borderOption;
    switch (borderType)
    {
    case cv::BORDER_REFLECT:     borderOption = FASTCV_BORDER_REFLECT;    break;
    case cv::BORDER_REFLECT_101: borderOption = FASTCV_BORDER_REFLECT_V2; break;
    case cv::BORDER_REPLICATE:   borderOption = FASTCV_BORDER_REPLICATE;  break;
    default:                     borderOption = FASTCV_BORDER_UNDEFINED;  break;
    }

    std::vector<fcvPyramidLevel_v2> lpyrSrc2(nLevels);

    int alignment = 8;
    if (useFloat)
    {
        // use version 2
        CV_Assert(fcvPyramidAllocate_v2(lpyrSrc2.data(), src.cols, src.rows, src.step, bytesPerPixel, nLevels, 0) == 0);
        CV_Assert(fcvPyramidCreatef32_v2((const float*)src.data, src.cols, src.rows, src.step, nLevels, lpyrSrc2.data()) == 0);
    }
    else
    {
        // use version 4
        fcvStatus statusAlloc = fcvPyramidAllocate_v3(lpyrSrc2.data(), src.cols, src.rows, src.step,
                                                      bytesPerPixel, alignment, nLevels, scaleOption, 0);
        if (statusAlloc != FASTCV_SUCCESS)
        {
            std::string s = fcvStatusStrings.count(statusAlloc) ? fcvStatusStrings.at(statusAlloc) : "unknown";
            CV_Error( cv::Error::StsInternal, "fcvPyramidAllocate_v3 error: " + s);
        }

        fcvStatus statusPyr = fcvPyramidCreateu8_v4(src.data, src.cols, src.rows, src.step, nLevels, scaleOption,
                                                    lpyrSrc2.data(), borderOption, borderValue);
        if (statusPyr != FASTCV_SUCCESS)
        {
            std::string s = fcvStatusStrings.count(statusPyr) ? fcvStatusStrings.at(statusPyr) : "unknown";
            CV_Error( cv::Error::StsInternal, "fcvPyramidCreateu8_v4 error: " + s);
        }
    }

    // create vector
    _pyr.create(nLevels, 1, src.type(), -1);
    for (int i = 0; i < nLevels; i++)
    {
        cv::Mat m = cv::Mat((uint32_t)lpyrSrc2[i].height, (uint32_t)lpyrSrc2[i].width,
                             src.type(), (void*)lpyrSrc2[i].ptr, (size_t)lpyrSrc2[i].stride);

        _pyr.create(m.size(), m.type(), i);
        m.copyTo(_pyr.getMat(i));
    }

    fcvPyramidDelete_v2(lpyrSrc2.data(), nLevels, 1);
}

} // namespace fastcv
} // namespace cv
