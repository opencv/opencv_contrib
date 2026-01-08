// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __OPENCV_WECHAT_QRCODE_IMGSOURCE_HPP__
#define __OPENCV_WECHAT_QRCODE_IMGSOURCE_HPP__
#include "zxing/common/bytematrix.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/luminance_source.hpp"
namespace cv {
namespace wechat_qrcode {
class ImgSource : public zxing::LuminanceSource {
private:
    typedef LuminanceSource Super;
    zxing::ArrayRef<char> _matrix;
    unsigned char* rgbs;
    unsigned char* luminances;
    int dataWidth;
    int dataHeight;
    int left;
    int top;
    void makeGray();
    void makeGrayReset();

    void arrayCopy(unsigned char* src, int inputOffset, char* dst, int outputOffset,
                   int length) const;


    ~ImgSource();

public:
    ImgSource(unsigned char* pixels, int width, int height);
    ImgSource(unsigned char* pixels, int width, int height, int left, int top, int cropWidth,
              int cropHeight, zxing::ErrorHandler& err_handler);

    static zxing::Ref<ImgSource> create(unsigned char* pixels, int width, int height);
    static zxing::Ref<ImgSource> create(unsigned char* pixels, int width, int height, int left,
                                        int top, int cropWidth, int cropHeight, zxing::ErrorHandler& err_handler);
    void reset(unsigned char* pixels, int width, int height);
    zxing::ArrayRef<char> getRow(int y, zxing::ArrayRef<char> row,
                                    zxing::ErrorHandler& err_handler) const override;
    zxing::ArrayRef<char> getMatrix() const override;
    zxing::Ref<zxing::ByteMatrix> getByteMatrix() const override;

    bool isCropSupported() const override;
    zxing::Ref<LuminanceSource> crop(int left, int top, int width, int height,
                                     zxing::ErrorHandler& err_handler) const override;

    bool isRotateSupported() const override;
    zxing::Ref<LuminanceSource> rotateCounterClockwise(
        zxing::ErrorHandler& err_handler) const override;

    int getMaxSize() { return dataHeight * dataWidth; }
};
}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_IMGSOURCE_HPP__
