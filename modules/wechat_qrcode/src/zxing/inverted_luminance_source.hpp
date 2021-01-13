// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_INVERTED_LUMINANCE_SOURCE_HPP__
#define __ZXING_INVERTED_LUMINANCE_SOURCE_HPP__

#include "zxing/common/bytematrix.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/luminance_source.hpp"
#include "zxing/zxing.hpp"

namespace zxing {

class InvertedLuminanceSource : public LuminanceSource {
private:
    typedef LuminanceSource Super;
    const Ref<LuminanceSource> delegate;

public:
    explicit InvertedLuminanceSource(Ref<LuminanceSource> const&);

    ArrayRef<char> getRow(int y, ArrayRef<char> row, ErrorHandler& err_handler) const override;
    ArrayRef<char> getMatrix() const override;
    Ref<ByteMatrix> getByteMatrix() const override;

    bool isCropSupported() const override;
    Ref<LuminanceSource> crop(int left, int top, int width, int height) const override;

    bool isRotateSupported() const override;

    virtual Ref<LuminanceSource> invert() const override;

    Ref<LuminanceSource> rotateCounterClockwise() const override;

    virtual void denoseLuminanceSource(int inter) override;
};

}  // namespace zxing

#endif  // __ZXING_INVERTED_LUMINANCE_SOURCE_HPP__
