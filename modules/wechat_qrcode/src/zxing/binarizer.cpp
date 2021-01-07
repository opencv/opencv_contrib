// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  binarizer.cpp
 *  zxing
 *
 *  Created by Ralf Kistner on 16/10/2009.
 *  Copyright 2008 ZXing authors All rights reserved.
 *  Modified by Lukasz Warchol on 02/02/2010.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "zxing/binarizer.hpp"

#include <iostream>

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
    // return source_->getWidth();
    return width;
}

int Binarizer::getHeight() const {
    // return source_->getHeight();
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
