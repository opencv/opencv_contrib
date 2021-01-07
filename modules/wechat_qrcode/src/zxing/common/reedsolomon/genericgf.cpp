// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  genericgf.cpp
 *  zxing
 *
 *  Created by Lukas Stabe on 13/02/2012.
 *  Copyright 2012 ZXing authors All rights reserved.
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

#include "zxing/common/reedsolomon/genericgf.hpp"
#include "zxing/common/illegal_argument_exception.hpp"
#include "zxing/common/reedsolomon/genericgfpoly.hpp"

#include <iostream>

using zxing::ErrorHandler;
using zxing::GenericGF;
using zxing::GenericGFPoly;
using zxing::Ref;

// Ref<GenericGF> GenericGF::AZTEC_DATA_12(new GenericGF(0x1069, 4096, 1,
// err_handler)); Ref<GenericGF> GenericGF::AZTEC_DATA_10(new GenericGF(0x409,
// 1024, 1, err_handler)); Ref<GenericGF> GenericGF::AZTEC_DATA_6(new
// GenericGF(0x43, 64, 1, err_handler )); Ref<GenericGF>
// GenericGF::AZTEC_PARAM(new GenericGF(0x13, 16, 1, err_handler));
// Ref<GenericGF> GenericGF::QR_CODE_FIELD_256(new GenericGF(0x011D, 256, 0,
// err_handler)); Ref<GenericGF> GenericGF::DATA_MATRIX_FIELD_256(new
// GenericGF(0x012D, 256, 1, err_handler)); Ref<GenericGF>
// GenericGF::AZTEC_DATA_8 = DATA_MATRIX_FIELD_256; Ref<GenericGF>
// GenericGF::MAXICODE_FIELD_64 = AZTEC_DATA_6; Ref<GenericGF> GenericGF::WXCODE
// = QR_CODE_FIELD_256;

// namespace {
//  int INITIALIZATION_THRESHOLD = 0;
//}

GenericGF::GenericGF(int primitive_, int size_, int b, ErrorHandler &err_handler)
    : size(size_), primitive(primitive_), generatorBase(b) {
    expTable.resize(size);
    logTable.resize(size);

    int x = 1;

    for (int i = 0; i < size; i++) {
        expTable[i] = x;
        x <<= 1;  // x = x * 2; we're assuming the generator alpha is 2
        if (x >= size) {
            x ^= primitive;
            x &= size - 1;
        }
    }
    for (int i = 0; i < size - 1; i++) {
        logTable[expTable[i]] = i;
    }
    // logTable[0] == 0 but this should never be used
    zero =
        Ref<GenericGFPoly>(new GenericGFPoly(*this, ArrayRef<int>(new Array<int>(1)), err_handler));
    zero->getCoefficients()[0] = 0;
    one =
        Ref<GenericGFPoly>(new GenericGFPoly(*this, ArrayRef<int>(new Array<int>(1)), err_handler));
    one->getCoefficients()[0] = 1;
    if (err_handler.ErrCode()) return;
    //  initialized = true;
}

// void GenericGF::checkInit() {
//  if (!initialized) {
//    initialize();
//  }
//}

Ref<GenericGFPoly> GenericGF::getZero() {
    //  checkInit();
    return zero;
}

Ref<GenericGFPoly> GenericGF::getOne() {
    //  checkInit();
    return one;
}

Ref<GenericGFPoly> GenericGF::buildMonomial(int degree, int coefficient,
                                            ErrorHandler &err_handler) {
    // checkInit();

    if (degree < 0) {
        err_handler = IllegalArgumentErrorHandler("Degree must be non-negative");
        return Ref<GenericGFPoly>();
    }
    if (coefficient == 0) {
        return zero;
    }
    ArrayRef<int> coefficients(new Array<int>(degree + 1));
    coefficients[0] = coefficient;

    Ref<GenericGFPoly> gfpoly(new GenericGFPoly(*this, coefficients, err_handler));
    if (err_handler.ErrCode()) return Ref<GenericGFPoly>();
    return gfpoly;
}

int GenericGF::addOrSubtract(int a, int b) { return a ^ b; }

int GenericGF::exp(int a) {
    // checkInit();
    return expTable[a];
}

int GenericGF::log(int a, ErrorHandler &err_handler) {
    // checkInit();
    if (a == 0) {
        err_handler = IllegalArgumentErrorHandler("cannot give log(0)");
        return -1;
    }
    return logTable[a];
}

int GenericGF::inverse(int a, ErrorHandler &err_handler) {
    // checkInit();
    if (a == 0) {
        err_handler = IllegalArgumentErrorHandler("Cannot calculate the inverse of 0");
        return -1;
    }
    return expTable[size - logTable[a] - 1];
}

int GenericGF::multiply(int a, int b) {
    // checkInit();

    if (a == 0 || b == 0) {
        return 0;
    }

    return expTable[(logTable[a] + logTable[b]) % (size - 1)];
}

int GenericGF::getSize() { return size; }

int GenericGF::getGeneratorBase() { return generatorBase; }
