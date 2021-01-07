// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#include "zxing/common/bytematrix.hpp"
#include <stdio.h>
#include "zxing/common/illegal_argument_exception.hpp"

#include <iostream>
#include <sstream>
#include <string>

using std::ostream;
using std::ostringstream;

using zxing::ArrayRef;
using zxing::ByteMatrix;
using zxing::ErrorHandler;
using zxing::Ref;

void ByteMatrix::init(int width, int height) {
    if (width < 1 || height < 1) {
        return;
    }
    this->width = width;
    this->height = height;
    bytes = new unsigned char[width * height];
    row_offsets = new int[height];
    row_offsets[0] = 0;
    for (int i = 1; i < height; i++) {
        row_offsets[i] = row_offsets[i - 1] + width;
    }
}

ByteMatrix::ByteMatrix(int dimension) { init(dimension, dimension); }

ByteMatrix::ByteMatrix(int width, int height) { init(width, height); }

ByteMatrix::ByteMatrix(int width, int height, ArrayRef<char> source) {
    init(width, height);
    int size = width * height;
    memcpy(&bytes[0], &source[0], size);
}

ByteMatrix::~ByteMatrix() {
    if (bytes) delete[] bytes;
    if (row_offsets) delete[] row_offsets;
}

unsigned char* ByteMatrix::getByteRow(int y, ErrorHandler& err_handler) {
    if (y < 0 || y >= getHeight()) {
        err_handler = IllegalArgumentErrorHandler("Requested row is outside the image.");
        return NULL;
    }
    return &bytes[row_offsets[y]];
}
