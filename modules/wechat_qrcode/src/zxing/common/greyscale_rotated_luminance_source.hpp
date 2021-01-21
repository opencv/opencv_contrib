// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_GREYSCALE_ROTATED_LUMINANCE_SOURCE_HPP__
#define __ZXING_COMMON_GREYSCALE_ROTATED_LUMINANCE_SOURCE_HPP__

#include "../errorhandler.hpp"
#include "../luminance_source.hpp"
#include "bytematrix.hpp"
namespace zxing {

class GreyscaleRotatedLuminanceSource : public LuminanceSource {
private:
    typedef LuminanceSource Super;
    ArrayRef<char> greyData_;
    const int dataWidth_;
    const int left_;
    const int top_;

public:
    GreyscaleRotatedLuminanceSource(ArrayRef<char> greyData, int dataWidth, int dataHeight,
                                    int left, int top, int _width, int _height,
                                    ErrorHandler& err_handler);

    ArrayRef<char> getRow(int y, ArrayRef<char> row, ErrorHandler& err_handler) const override;
    ArrayRef<char> getMatrix() const override;
    Ref<ByteMatrix> getByteMatrix() const override;
};

}  // namespace zxing

#endif  // __ZXING_COMMON_GREYSCALE_ROTATED_LUMINANCE_SOURCE_HPP__
