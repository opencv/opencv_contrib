// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../../precomp.hpp"
#include "genericgf.hpp"
#include "genericgfpoly.hpp"

using zxing::ErrorHandler;
using zxing::GenericGF;
using zxing::GenericGFPoly;
using zxing::Ref;

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

Ref<GenericGFPoly> GenericGF::getZero() { return zero; }

Ref<GenericGFPoly> GenericGF::getOne() { return one; }

Ref<GenericGFPoly> GenericGF::buildMonomial(int degree, int coefficient,
                                            ErrorHandler &err_handler) {
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

int GenericGF::exp(int a) { return expTable[a]; }

int GenericGF::log(int a, ErrorHandler &err_handler) {
    if (a == 0) {
        err_handler = IllegalArgumentErrorHandler("cannot give log(0)");
        return -1;
    }
    return logTable[a];
}

int GenericGF::inverse(int a, ErrorHandler &err_handler) {
    if (a == 0) {
        err_handler = IllegalArgumentErrorHandler("Cannot calculate the inverse of 0");
        return -1;
    }
    return expTable[size - logTable[a] - 1];
}

int GenericGF::multiply(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }

    return expTable[(logTable[a] + logTable[b]) % (size - 1)];
}

int GenericGF::getSize() { return size; }

int GenericGF::getGeneratorBase() { return generatorBase; }
