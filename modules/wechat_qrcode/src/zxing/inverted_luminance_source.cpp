// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#include "zxing/inverted_luminance_source.hpp"

using zxing::ArrayRef;
using zxing::boolean;
using zxing::ByteMatrix;
using zxing::ErrorHandler;
using zxing::InvertedLuminanceSource;
using zxing::LuminanceSource;
using zxing::Ref;

InvertedLuminanceSource::InvertedLuminanceSource(Ref<LuminanceSource> const& delegate_)
    : Super(delegate_->getWidth(), delegate_->getHeight()), delegate(delegate_) {}

ArrayRef<char> InvertedLuminanceSource::getRow(int y, ArrayRef<char> row,
                                               ErrorHandler& err_handler) const {
    row = delegate->getRow(y, row, err_handler);
    if (err_handler.ErrCode()) return ArrayRef<char>();
    int width = getWidth();
    for (int i = 0; i < width; i++) {
        row[i] = (byte)(255 - (row[i] & 0xFF));
    }
    return row;
}

ArrayRef<char> InvertedLuminanceSource::getMatrix() const {
    ArrayRef<char> matrix = delegate->getMatrix();
    int length = getWidth() * getHeight();
    ArrayRef<char> invertedMatrix(length);
    for (int i = 0; i < length; i++) {
        invertedMatrix[i] = (byte)(255 - (matrix[i] & 0xFF));
    }
    return invertedMatrix;
}

bool InvertedLuminanceSource::isCropSupported() const { return delegate->isCropSupported(); }

Ref<LuminanceSource> InvertedLuminanceSource::crop(int left, int top, int width, int height) const {
    return Ref<LuminanceSource>(
        new InvertedLuminanceSource(delegate->crop(left, top, width, height)));
}

bool InvertedLuminanceSource::isRotateSupported() const { return delegate->isRotateSupported(); }

Ref<LuminanceSource> InvertedLuminanceSource::invert() const { return delegate; }

Ref<LuminanceSource> InvertedLuminanceSource::rotateCounterClockwise() const {
    return Ref<LuminanceSource>(new InvertedLuminanceSource(delegate->rotateCounterClockwise()));
}

void InvertedLuminanceSource::denoseLuminanceSource(int inter) { tvInter = inter; }

Ref<ByteMatrix> InvertedLuminanceSource::getByteMatrix() const {
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
