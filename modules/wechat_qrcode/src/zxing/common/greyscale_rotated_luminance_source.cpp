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
#include "greyscale_rotated_luminance_source.hpp"
#include "bytematrix.hpp"
using zxing::ArrayRef;
using zxing::ByteMatrix;
using zxing::ErrorHandler;
using zxing::GreyscaleRotatedLuminanceSource;
using zxing::Ref;

// Note that dataWidth and dataHeight are not reversed, as we need to
// be able to traverse the greyData correctly, which does not get
// rotated.
GreyscaleRotatedLuminanceSource::GreyscaleRotatedLuminanceSource(ArrayRef<char> greyData,
                                                                 int dataWidth, int dataHeight,
                                                                 int left, int top, int _width,
                                                                 int _height,
                                                                 ErrorHandler& err_handler)
    : Super(_width, _height), greyData_(greyData), dataWidth_(dataWidth), left_(left), top_(top) {
    // Intentionally comparing to the opposite dimension since we're rotated.
    if (left + _width > dataHeight || top + _height > dataWidth) {
        err_handler = IllegalArgumentErrorHandler("Crop rectangle does not fit within image data.");
    }
}

// The API asks for rows, but we're rotated, so we return columns.
ArrayRef<char> GreyscaleRotatedLuminanceSource::getRow(int y, ArrayRef<char> row,
                                                       ErrorHandler& err_handler) const {
    if (y < 0 || y >= getHeight()) {
        err_handler = IllegalArgumentErrorHandler("Requested row is outside the image.");
        return ArrayRef<char>();
    }
    if (!row || row->size() < getWidth()) {
        row = ArrayRef<char>(getWidth());
    }
    int offset = (left_ * dataWidth_) + (dataWidth_ - 1 - (y + top_));
    for (int x = 0; x < getWidth(); x++) {
        row[x] = greyData_[offset];
        offset += dataWidth_;
    }
    return row;
}

ArrayRef<char> GreyscaleRotatedLuminanceSource::getMatrix() const {
    ArrayRef<char> result(getWidth() * getHeight());
    for (int y = 0; y < getHeight(); y++) {
        char* row = &result[y * getWidth()];
        int offset = (left_ * dataWidth_) + (dataWidth_ - 1 - (y + top_));
        for (int x = 0; x < getWidth(); x++) {
            row[x] = greyData_[offset];
            offset += dataWidth_;
        }
    }
    return result;
}

Ref<ByteMatrix> GreyscaleRotatedLuminanceSource::getByteMatrix() const {
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
