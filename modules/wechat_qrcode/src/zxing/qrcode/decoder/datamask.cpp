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
#include "datamask.hpp"
namespace zxing {
namespace qrcode {

using zxing::ErrorHandler;

DataMask::DataMask() {}

DataMask::~DataMask() {}

DataMask& DataMask::forReference(int reference, ErrorHandler& err_handler) {
    if (reference < 0 || reference > 7) {
        err_handler = zxing::IllegalArgumentErrorHandler("reference must be between 0 and 7");
        return *DATA_MASKS[0];
    }
    return *DATA_MASKS[reference];
}

void DataMask::unmaskBitMatrix(BitMatrix& bits, size_t dimension) {
    for (size_t y = 0; y < dimension; y++) {
        for (size_t x = 0; x < dimension; x++) {
            // TODO: check why the coordinates have to be swapped
            if (isMasked(y, x)) {
                bits.flip(x, y);
            }
        }
    }
}

/**
 * 000: mask bits for which (x + y) mod 2 == 0
 */
class DataMask000 : public DataMask {
public:
    bool isMasked(size_t x, size_t y) override { return ((x + y) % 2) == 0; }
};

/**
 * 001: mask bits for which x mod 2 == 0
 */
class DataMask001 : public DataMask {
public:
    bool isMasked(size_t x, size_t) override { return (x % 2) == 0; }
};

/**
 * 010: mask bits for which y mod 3 == 0
 */
class DataMask010 : public DataMask {
public:
    bool isMasked(size_t, size_t y) override { return y % 3 == 0; }
};

/**
 * 011: mask bits for which (x + y) mod 3 == 0
 */
class DataMask011 : public DataMask {
public:
    bool isMasked(size_t x, size_t y) override { return (x + y) % 3 == 0; }
};

/**
 * 100: mask bits for which (x/2 + y/3) mod 2 == 0
 */
class DataMask100 : public DataMask {
public:
    bool isMasked(size_t x, size_t y) override { return (((x >> 1) + (y / 3)) % 2) == 0; }
};

/**
 * 101: mask bits for which xy mod 2 + xy mod 3 == 0
 */
class DataMask101 : public DataMask {
public:
    bool isMasked(size_t x, size_t y) override {
        size_t temp = x * y;
        return (temp % 2) + (temp % 3) == 0;
    }
};

/**
 * 110: mask bits for which (xy mod 2 + xy mod 3) mod 2 == 0
 */
class DataMask110 : public DataMask {
public:
    bool isMasked(size_t x, size_t y) override {
        size_t temp = x * y;
        return (((temp % 2) + (temp % 3)) % 2) == 0;
    }
};

/**
 * 111: mask bits for which ((x+y)mod 2 + xy mod 3) mod 2 == 0
 */
class DataMask111 : public DataMask {
public:
    bool isMasked(size_t x, size_t y) override {
        return ((((x + y) % 2) + ((x * y) % 3)) % 2) == 0;
    }
};

vector<Ref<DataMask> > DataMask::DATA_MASKS = {
    Ref<DataMask>(new DataMask000()), Ref<DataMask>(new DataMask001()),
    Ref<DataMask>(new DataMask010()), Ref<DataMask>(new DataMask011()),
    Ref<DataMask>(new DataMask100()), Ref<DataMask>(new DataMask101()),
    Ref<DataMask>(new DataMask110()), Ref<DataMask>(new DataMask111()),
};

}  // namespace qrcode
}  // namespace zxing
