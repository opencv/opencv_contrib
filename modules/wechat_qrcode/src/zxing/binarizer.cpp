// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../precomp.hpp"
#include "binarizer.hpp"

namespace zxing {

Binarizer::Binarizer(Ref<LuminanceSource> source) : source_(source) {
    dataWidth = source->getWidth();
    dataHeight = source->getHeight();

    width = dataWidth;
    height = dataHeight;

    matrix_ = NULL;
    matrix0_ = NULL;
    matrixInverted_ = NULL;

    histogramBinarized = false;
    usingHistogram = false;
}

Binarizer::~Binarizer() {}

Ref<LuminanceSource> Binarizer::getLuminanceSource() const { return source_; }

int Binarizer::getWidth() const {
    return width;
}

int Binarizer::getHeight() const {
    return height;
}

int Binarizer::rotateCounterClockwise() { return 0; }

int Binarizer::rotateCounterClockwise45() { return 0; }

Ref<BitMatrix> Binarizer::getInvertedMatrix(ErrorHandler& err_handler) {
    if (!matrix_) {
        return Ref<BitMatrix>();
    }

    if (matrixInverted_ == NULL) {
        matrixInverted_ = new BitMatrix(matrix_->getWidth(), matrix_->getHeight(), err_handler);
        matrixInverted_->copyOf(matrix_, err_handler);
        matrixInverted_->flipAll();
    }

    return matrixInverted_;
}

// Return different black matrix according to cacheMode
Ref<BitMatrix> Binarizer::getBlackMatrix(ErrorHandler& err_handler) {
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    matrix_ = matrix0_;
    return matrix_;
}

Ref<BitArray> Binarizer::getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler) {
    if (!matrix_) {
        matrix_ = getBlackMatrix(err_handler);
        if (err_handler.ErrCode()) return Ref<BitArray>();
    }

    matrix_->getRow(y, row);
    return row;
}

ArrayRef<BINARIZER_BLOCK> Binarizer::getBlockArray(int size) {
    ArrayRef<BINARIZER_BLOCK> blocks(new Array<BINARIZER_BLOCK>(size));

    for (int i = 0; i < blocks->size(); i++) {
        blocks[i].sum = 0;
        blocks[i].min = 0xFF;
        blocks[i].max = 0;
    }

    return blocks;
}
}  // namespace zxing
