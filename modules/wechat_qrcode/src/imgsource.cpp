// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "precomp.hpp"
#include "imgsource.hpp"

using zxing::ArrayRef;
using zxing::ByteMatrix;
using zxing::ErrorHandler;
using zxing::LuminanceSource;
using zxing::Ref;
namespace cv {
namespace wechat_qrcode {

// Initialize the ImgSource
ImgSource::ImgSource(unsigned char* pixels, int width, int height)
    : Super(width, height) {
    luminances = new unsigned char[width * height];
    memset(luminances, 0, width * height);

    rgbs = pixels;

    dataWidth = width;
    dataHeight = height;
    left = 0;
    top = 0;

    // Make gray luminances first
    makeGray();
}

// Added for crop function
ImgSource::ImgSource(unsigned char* pixels, int width, int height, int left_, int top_,
                     int cropWidth, int cropHeight,
                     ErrorHandler& err_handler)
    : Super(cropWidth, cropHeight) {
    rgbs = pixels;

    dataWidth = width;
    dataHeight = height;
    left = left_;
    top = top_;

    // super(width, height);
    if ((left_ + cropWidth) > dataWidth || (top_ + cropHeight) > dataHeight || top_ < 0 ||
        left_ < 0) {
        err_handler =
            zxing::IllegalArgumentErrorHandler("Crop rectangle does not fit within image data.");
        return;
    }

    luminances = new unsigned char[width * height];

    // Make gray luminances first
    makeGray();
}

ImgSource::~ImgSource() {
    if (luminances != NULL) {
        delete[] luminances;
    }
}

Ref<ImgSource> ImgSource::create(unsigned char* pixels, int width, int height) {
    return Ref<ImgSource>(new ImgSource(pixels, width, height));
}

Ref<ImgSource> ImgSource::create(unsigned char* pixels, int width, int height, int left, int top,
                                 int cropWidth, int cropHeight,
                                 zxing::ErrorHandler& err_handler) {
    return Ref<ImgSource>(new ImgSource(pixels, width, height, left, top, cropWidth, cropHeight, err_handler));
}

void ImgSource::reset(unsigned char* pixels, int width, int height) {
    rgbs = pixels;
    left = 0;
    top = 0;

    setWidth(width);
    setHeight(height);
    dataWidth = width;
    dataHeight = height;
    makeGrayReset();
}

ArrayRef<char> ImgSource::getRow(int y, zxing::ArrayRef<char> row,
                                 zxing::ErrorHandler& err_handler) const {
    if (y < 0 || y >= getHeight()) {
        err_handler = zxing::IllegalArgumentErrorHandler("Requested row is outside the image");
        return ArrayRef<char>();
    }

    int width = getWidth();
    if (row->data() == NULL || row->empty() || row->size() < width) {
        row = zxing::ArrayRef<char>(width);
    }
    int offset = (y + top) * dataWidth + left;

    char* rowPtr = &row[0];
    arrayCopy(luminances, offset, rowPtr, 0, width);

    return row;
}

/** This is a more efficient implementation. */
ArrayRef<char> ImgSource::getMatrix() const {
    int width = getWidth();
    int height = getHeight();

    int area = width * height;

    // If the caller asks for the entire underlying image, save the copy and
    // give them the original data. The docs specifically warn that
    // result.length must be ignored.
    if (width == dataWidth && height == dataHeight) {
        return _matrix;
    }

    zxing::ArrayRef<char> newMatrix = zxing::ArrayRef<char>(area);

    int inputOffset = top * dataWidth + left;

    // If the width matches the full width of the underlying data, perform a
    // single copy.
    if (width == dataWidth) {
        arrayCopy(luminances, inputOffset, &newMatrix[0], 0, area);
        return newMatrix;
    }

    // Otherwise copy one cropped row at a time.
    for (int y = 0; y < height; y++) {
        int outputOffset = y * width;
        arrayCopy(luminances, inputOffset, &newMatrix[0], outputOffset, width);
        inputOffset += dataWidth;
    }
    return newMatrix;
}


void ImgSource::makeGray() {
    int area = dataWidth * dataHeight;
    _matrix = zxing::ArrayRef<char>(area);
    arrayCopy(rgbs, 0, &_matrix[0], 0, area);
}

void ImgSource::makeGrayReset() {
    int area = dataWidth * dataHeight;
    arrayCopy(rgbs, 0, &_matrix[0], 0, area);
}

void ImgSource::arrayCopy(unsigned char* src, int inputOffset, char* dst, int outputOffset,
                          int length) const {
    const unsigned char* srcPtr = src + inputOffset;
    char* dstPtr = dst + outputOffset;

    memcpy(dstPtr, srcPtr, length * sizeof(unsigned char));
}

bool ImgSource::isCropSupported() const { return true; }

Ref<LuminanceSource> ImgSource::crop(int left_, int top_, int width, int height,
                                     ErrorHandler& err_handler) const {
    return ImgSource::create(rgbs, dataWidth, dataHeight, left + left_, top + top_, width, height, err_handler);
}

bool ImgSource::isRotateSupported() const { return false; }

Ref<LuminanceSource> ImgSource::rotateCounterClockwise(ErrorHandler& err_handler) const {
    // Intentionally flip the left, top, width, and height arguments as
    // needed. dataWidth and dataHeight are always kept unrotated.
    int width = getWidth();
    int height = getHeight();

    return ImgSource::create(rgbs, dataWidth, dataHeight, top, left, height, width, err_handler);
}


Ref<ByteMatrix> ImgSource::getByteMatrix() const {
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
}  // namespace wechat_qrcode
}  // namespace cv