// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../precomp.hpp"
#include "greyscale_luminance_source.hpp"
#include "bytematrix.hpp"
#include "greyscale_rotated_luminance_source.hpp"
using zxing::ArrayRef;
using zxing::ByteMatrix;
using zxing::ErrorHandler;
using zxing::GreyscaleLuminanceSource;
using zxing::LuminanceSource;
using zxing::Ref;

GreyscaleLuminanceSource::GreyscaleLuminanceSource(ArrayRef<char> greyData, int dataWidth,
                                                   int dataHeight, int left, int top, int width,
                                                   int height, ErrorHandler& err_handler)
    : Super(width, height),
      greyData_(greyData),
      dataWidth_(dataWidth),
      dataHeight_(dataHeight),
      left_(left),
      top_(top) {
    if (left + width > dataWidth || top + height > dataHeight || top < 0 || left < 0) {
        err_handler = IllegalArgumentErrorHandler("Crop rectangle does not fit within image data.");
    }
}

ArrayRef<char> GreyscaleLuminanceSource::getRow(int y, ArrayRef<char> row,
                                                ErrorHandler& err_handler) const {
    if (y < 0 || y >= this->getHeight()) {
        err_handler = IllegalArgumentErrorHandler("Requested row is outside the image.");
        return ArrayRef<char>();
    }
    int width = getWidth();
    if (!row || row->size() < width) {
        ArrayRef<char> temp(width);
        row = temp;
    }
    int offset = (y + top_) * dataWidth_ + left_;
    memcpy(&row[0], &greyData_[offset], width);
    return row;
}

ArrayRef<char> GreyscaleLuminanceSource::getMatrix() const {
    int size = getWidth() * getHeight();
    ArrayRef<char> result(size);
    if (left_ == 0 && top_ == 0 && dataWidth_ == getWidth() && dataHeight_ == getHeight()) {
        memcpy(&result[0], &greyData_[0], size);
    } else {
        for (int row = 0; row < getHeight(); row++) {
            memcpy(&result[row * getWidth()], &greyData_[(top_ + row) * dataWidth_ + left_],
                   getWidth());
        }
    }
    return result;
}

Ref<LuminanceSource> GreyscaleLuminanceSource::rotateCounterClockwise(
    ErrorHandler& err_handler) const {
    // Intentionally flip the left, top, width, and height arguments as
    // needed. dataWidth and dataHeight are always kept unrotated.
    Ref<LuminanceSource> result(new GreyscaleRotatedLuminanceSource(
        greyData_, dataWidth_, dataHeight_, top_, left_, getHeight(), getWidth(), err_handler));
    if (err_handler.ErrCode()) return Ref<LuminanceSource>();
    return result;
}

Ref<ByteMatrix> GreyscaleLuminanceSource::getByteMatrix() const {
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
