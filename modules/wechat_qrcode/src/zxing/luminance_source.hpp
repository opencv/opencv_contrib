// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_LUMINANCE_SOURCE_HPP__
#define __ZXING_LUMINANCE_SOURCE_HPP__

#include <string.h>
#include "common/array.hpp"
#include "common/bytematrix.hpp"
#include "common/counted.hpp"
#include "errorhandler.hpp"

namespace zxing {

class LuminanceSource : public Counted {
protected:
    int width_;
    int height_;

public:
    LuminanceSource(int width, int height);
    virtual ~LuminanceSource();

    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    void setWidth(int w) { width_ = w; }
    void setHeight(int h) { height_ = h; }
    void filter();

    // Callers take ownership of the returned memory and must call delete [] on
    // it themselves.
    virtual ArrayRef<char> getRow(int y, ArrayRef<char> row,
                                  zxing::ErrorHandler& err_handler) const = 0;
    virtual ArrayRef<char> getMatrix() const = 0;
    virtual Ref<ByteMatrix> getByteMatrix() const = 0;

    virtual bool isCropSupported() const;
    virtual Ref<LuminanceSource> crop(int left, int top, int width, int height,
                                      zxing::ErrorHandler& err_handler) const;

    virtual bool isRotateSupported() const;

    virtual Ref<LuminanceSource> rotateCounterClockwise(zxing::ErrorHandler& err_handler) const;

    operator std::string() const;
};

}  // namespace zxing

#endif  // __ZXING_LUMINANCE_SOURCE_HPP__
