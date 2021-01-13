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
    int _comps;
    int _pixelStep;
    zxing::ArrayRef<char> _matrix;
    unsigned char* rgbs;
    unsigned char* luminances;
    int dataWidth;
    int dataHeight;
    int maxDataWidth;
    int maxDataHeight;
    int left;
    int top;

    unsigned char convertPixel(unsigned char const* pixel, zxing::ErrorHandler& err_handler) const;
    void makeGrayRow(int y, zxing::ErrorHandler& err_handler);
    void makeGray(zxing::ErrorHandler& err_handler);
    void makeGrayReset(zxing::ErrorHandler& err_handler);

    void arrayCopy(unsigned char* src, int inputOffset, char* dst, int outputOffset,
                   int length) const;

    zxing::ArrayRef<char> downSample(zxing::ArrayRef<char> image, int& width, int& height,
                                     int pixelStep);

    ~ImgSource();

public:
    ImgSource(unsigned char* pixels, int width, int height, int comps, int pixelStep,
              zxing::ErrorHandler& err_handler);
    ImgSource(unsigned char* pixels, int width, int height, int left, int top, int cropWidth,
              int cropHeight, int comps, int pixelStep, zxing::ErrorHandler& err_handler);

    static zxing::Ref<ImgSource> create(unsigned char* pixels, int width, int height, int comps,
                                        int pixelStep, zxing::ErrorHandler& err_handler);
    static zxing::Ref<ImgSource> create(unsigned char* pixels, int width, int height, int left,
                                        int top, int cropWidth, int cropHeight, int comps,
                                        int pixelStep, zxing::ErrorHandler& err_handler);

    void reset(unsigned char* pixels, int width, int height, int comps, int pixelStep,
               zxing::ErrorHandler& err_handler);

    zxing::ArrayRef<char> getRow(int y, zxing::ArrayRef<char> row,
                                 zxing::ErrorHandler& err_handler) const override;
    zxing::ArrayRef<char> getMatrix() const override;
    zxing::Ref<zxing::ByteMatrix> getByteMatrix() const override;
    int tvInter;
    virtual void denoseLuminanceSource(int inter) override;
    void tvDenoising() const;

    bool isCropSupported() const override;
    zxing::Ref<LuminanceSource> crop(int left, int top, int width, int height,
                                     zxing::ErrorHandler& err_handler) const;

    bool isRotateSupported() const override;
    zxing::Ref<LuminanceSource> rotateCounterClockwise(zxing::ErrorHandler& err_handler) const;

    int getMaxSize() { return maxDataHeight * maxDataWidth; }
};
}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_IMGSOURCE_HPP__
