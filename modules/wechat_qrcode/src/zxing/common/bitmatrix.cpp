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
#include "bitmatrix.hpp"

using zxing::ArrayRef;
using zxing::BitArray;
using zxing::BitMatrix;
using zxing::ErrorHandler;
using zxing::Ref;

void BitMatrix::init(int _width, int _height, ErrorHandler& err_handler) {
    if (_width < 1 || _height < 1) {
        err_handler = IllegalArgumentErrorHandler("Both dimensions must be greater than 0");
        return;
    }
    width = _width;
    height = _height;
    this->rowBitsSize = width;
    bits = ArrayRef<unsigned char>(width * height);
    rowOffsets = ArrayRef<int>(height);

    // offsetRowSize = new int[height];
    rowOffsets[0] = 0;
    for (int i = 1; i < height; i++) {
        rowOffsets[i] = rowOffsets[i - 1] + width;
    }

    isInitRowCounters = false;
    isInitColsCounters = false;
}

void BitMatrix::init(int _width, int _height, unsigned char* bitsPtr, ErrorHandler& err_handler) {
    init(_width, _height, err_handler);
    if (err_handler.ErrCode()) return;
    memcpy(bits->data(), bitsPtr, width * height * sizeof(unsigned char));
}

void BitMatrix::initRowCounters() {
    if (isInitRowCounters == true) {
        return;
    }

    row_counters = vector<COUNTER_TYPE>(width * height, 0);
    row_counters_offset = vector<COUNTER_TYPE>(width * height, 0);
    row_point_offset = vector<COUNTER_TYPE>(width * height, 0);
    row_counter_offset_end = vector<COUNTER_TYPE>(height, 0);

    row_counters_recorded = vector<bool>(height, false);

    isInitRowCounters = true;
}
void BitMatrix::initColsCounters() {
    if (isInitColsCounters == true) {
        return;
    }

    cols_counters = vector<COUNTER_TYPE>(width * height, 0);
    cols_counters_offset = vector<COUNTER_TYPE>(width * height, 0);
    cols_point_offset = vector<COUNTER_TYPE>(width * height, 0);
    cols_counter_offset_end = vector<COUNTER_TYPE>(width, 0);

    cols_counters_recorded = vector<bool>(width, false);

    isInitColsCounters = true;
}

BitMatrix::BitMatrix(int dimension, ErrorHandler& err_handler) {
    init(dimension, dimension, err_handler);
}

BitMatrix::BitMatrix(int _width, int _height, ErrorHandler& err_handler) {
    init(_width, _height, err_handler);
}

BitMatrix::BitMatrix(int _width, int _height, unsigned char* bitsPtr, ErrorHandler& err_handler) {
    init(_width, _height, bitsPtr, err_handler);
}
// Copy bitMatrix
void BitMatrix::copyOf(Ref<BitMatrix> _bits, ErrorHandler& err_handler) {
    int _width = _bits->getWidth();
    int _height = _bits->getHeight();
    init(_width, _height, err_handler);

    for (int y = 0; y < height; y++) {
        bool* rowPtr = _bits->getRowBoolPtr(y);
        setRowBool(y, rowPtr);
    }
}

void BitMatrix::xxor(Ref<BitMatrix> _bits) {
    if (width != _bits->getWidth() || height != _bits->getHeight()) {
        return;
    }

    for (int y = 0; y < height && y < _bits->getHeight(); ++y) {
        bool* rowPtrA = _bits->getRowBoolPtr(y);
        bool* rowPtrB = getRowBoolPtr(y);

        for (int x = 0; x < width && x < _bits->getWidth(); ++x) {
            rowPtrB[x] = rowPtrB[x] ^ rowPtrA[x];
        }
        setRowBool(y, rowPtrB);
    }
}

BitMatrix::~BitMatrix() {}

void BitMatrix::flip(int x, int y) {
    bits[rowOffsets[y] + x] = (bits[rowOffsets[y] + x] == (unsigned char)0);
}

void BitMatrix::flipAll() {
    bool* matrixBits = (bool*)bits->data();
    for (int i = 0; i < bits->size(); i++) {
        matrixBits[i] = !matrixBits[i];
    }
}

void BitMatrix::flipRegion(int left, int top, int _width, int _height, ErrorHandler& err_handler) {
    if (top < 0 || left < 0) {
        err_handler = IllegalArgumentErrorHandler("Left and top must be nonnegative");
        return;
    }
    if (_height < 1 || _width < 1) {
        err_handler = IllegalArgumentErrorHandler("Height and width must be at least 1");
        return;
    }
    int right = left + _width;
    int bottom = top + _height;
    if (bottom > this->height || right > this->width) {
        err_handler = IllegalArgumentErrorHandler("The region must fit inside the matrix");
        return;
    }

    for (int y = top; y < bottom; y++) {
        for (int x = left; x < right; x++) {
            bits[rowOffsets[y] + x] ^= 1;
        }
    }
}

void BitMatrix::setRegion(int left, int top, int _width, int _height, ErrorHandler& err_handler) {
    if (top < 0 || left < 0) {
        err_handler = IllegalArgumentErrorHandler("Left and top must be nonnegative");
        return;
    }
    if (_height < 1 || _width < 1) {
        err_handler = IllegalArgumentErrorHandler("Height and width must be at least 1");
        return;
    }
    int right = left + _width;
    int bottom = top + _height;
    if (bottom > this->height || right > this->width) {
        err_handler = IllegalArgumentErrorHandler("The region must fit inside the matrix");
        return;
    }

    for (int y = top; y < bottom; y++) {
        for (int x = left; x < right; x++) {
            bits[rowOffsets[y] + x] = true;
            // bits[rowOffsets[y]+x] |= 0xFF;
        }
    }
}

Ref<BitArray> BitMatrix::getRow(int y, Ref<BitArray> row) {
    if (row.empty() || row->getSize() < width) {
        row = new BitArray(width);
    }

    // row->
    unsigned char* src = bits.data() + rowOffsets[y];
    row->setOneRow(src, width);

    return row;
}

ArrayRef<int> BitMatrix::getTopLeftOnBit() const {
    int bitsOffset = 0;
    while (bitsOffset < bits->size() && bits[bitsOffset] == 0) {
        bitsOffset++;
    }
    if (bitsOffset == bits->size()) {
        return ArrayRef<int>();
    }
    int y = bitsOffset / width;
    int x = bitsOffset % width;
    ArrayRef<int> res(2);
    res[0] = x;
    res[1] = y;
    return res;
}

ArrayRef<int> BitMatrix::getBottomRightOnBit() const {
    int bitsOffset = bits->size() - 1;
    while (bitsOffset >= 0 && bits[bitsOffset] == 0) {
        bitsOffset--;
    }
    if (bitsOffset < 0) {
        return ArrayRef<int>();
    }

    int y = bitsOffset / width;
    int x = bitsOffset % width;
    ArrayRef<int> res(2);
    res[0] = x;
    res[1] = y;
    return res;
}

void BitMatrix::getRowBool(int y, bool* getrow) {
    int offset = rowOffsets[y];
    unsigned char* src = bits.data() + offset;
    memcpy(getrow, src, rowBitsSize * sizeof(bool));
}

void BitMatrix::setRowBool(int y, bool* row) {
    int offset = rowOffsets[y];
    unsigned char* dst = bits.data() + offset;
    memcpy(dst, row, rowBitsSize * sizeof(bool));

    return;
}

bool* BitMatrix::getRowBoolPtr(int y) {
    int offset = y * rowBitsSize;
    unsigned char* src = bits.data() + offset;
    return (bool*)src;
}

void BitMatrix::clear() {
    int size = bits->size();

    unsigned char* dst = bits->data();
    memset(dst, 0, size * sizeof(unsigned char));
}

int BitMatrix::getWidth() const { return width; }

int BitMatrix::getHeight() const { return height; }

COUNTER_TYPE* BitMatrix::getRowPointInRecords(int y) {
    if (!row_point_offset[y]) {
        setRowRecords(y);
    }
    int offset = y * width;
    COUNTER_TYPE* counters = &row_point_offset[0] + offset;
    return (COUNTER_TYPE*)counters;
}

COUNTER_TYPE* BitMatrix::getRowRecords(int y) {
    if (!row_counters_recorded[y]) {
        setRowRecords(y);
    }
    int offset = y * width;
    COUNTER_TYPE* counters = &row_counters[0] + offset;
    return (COUNTER_TYPE*)counters;
}

COUNTER_TYPE* BitMatrix::getRowRecordsOffset(int y) {
    if (!row_counters_recorded[y]) {
        setRowRecords(y);
    }
    int offset = y * width;
    COUNTER_TYPE* counters = &row_counters_offset[0] + offset;
    return (COUNTER_TYPE*)counters;
}

bool BitMatrix::getRowFirstIsWhite(int y) {
    bool is_white = !get(0, y);
    return is_white;
}

bool BitMatrix::getRowLastIsWhite(int y) {
    bool last_is_white = !get(width - 1, y);
    return last_is_white;
}

COUNTER_TYPE BitMatrix::getRowCounterOffsetEnd(int y) {
    if (!row_counters_recorded[y]) {
        setRowRecords(y);
    }
    return row_counter_offset_end[y];
}

void BitMatrix::setRowRecords(int y) {
    COUNTER_TYPE* cur_row_counters = &row_counters[0] + y * width;
    COUNTER_TYPE* cur_row_counters_offset = &row_counters_offset[0] + y * width;
    COUNTER_TYPE* cur_row_point_in_counters = &row_point_offset[0] + y * width;
    int end = width;

    bool* rowBit = getRowBoolPtr(y);
    bool isWhite = !rowBit[0];
    int counterPosition = 0;
    int i = 0;
    cur_row_counters_offset[0] = 0;
    while (i < end) {
        if (rowBit[i] ^ isWhite) {  // that is, exactly one is true
            cur_row_counters[counterPosition]++;
        } else {
            counterPosition++;
            if (counterPosition == end) {
                break;
            } else {
                cur_row_counters[counterPosition] = 1;
                isWhite = !isWhite;
                cur_row_counters_offset[counterPosition] = i;
            }
        }
        cur_row_point_in_counters[i] = counterPosition;
        i++;
    }

    // use the last row__onedReaderData->counter_size to record
    // _onedReaderData->counter_size
    row_counter_offset_end[y] = counterPosition < end ? (counterPosition + 1) : end;

    row_counters_recorded[y] = true;
    return;
}

COUNTER_TYPE* BitMatrix::getColsPointInRecords(int x) {
    if (!cols_point_offset[x]) {
        setColsRecords(x);
    }
    int offset = x * height;
    COUNTER_TYPE* counters = &cols_point_offset[0] + offset;
    return (COUNTER_TYPE*)counters;
}

COUNTER_TYPE* BitMatrix::getColsRecords(int x) {
    if (!cols_counters_recorded[x]) {
        setColsRecords(x);
    }
    int offset = x * height;
    COUNTER_TYPE* counters = &cols_counters[0] + offset;
    return (COUNTER_TYPE*)counters;
}

COUNTER_TYPE* BitMatrix::getColsRecordsOffset(int x) {
    if (!cols_counters_recorded[x]) {
        setColsRecords(x);
    }
    int offset = x * height;
    COUNTER_TYPE* counters = &cols_counters_offset[0] + offset;
    return (COUNTER_TYPE*)counters;
}

COUNTER_TYPE BitMatrix::getColsCounterOffsetEnd(int x) {
    if (!cols_counters_recorded[x]) {
        setColsRecords(x);
    }
    return cols_counter_offset_end[x];
}

void BitMatrix::setColsRecords(int x) {
    COUNTER_TYPE* cur_cols_counters = &cols_counters[0] + x * height;
    COUNTER_TYPE* cur_cols_counters_offset = &cols_counters_offset[0] + x * height;
    COUNTER_TYPE* cur_cols_point_in_counters = &cols_point_offset[0] + x * height;
    int end = height;

    bool* rowBit = getRowBoolPtr(0);
    bool isWhite = !rowBit[0];
    int counterPosition = 0;
    int i = 0;
    cur_cols_counters_offset[0] = 0;
    while (i < end) {
        if (rowBit[i] ^ isWhite) {  // that is, exactly one is true
            cur_cols_counters[counterPosition]++;
        } else {
            counterPosition++;
            if (counterPosition == end) {
                break;
            } else {
                cur_cols_counters[counterPosition] = 1;
                isWhite = !isWhite;
                cur_cols_counters_offset[counterPosition] = i;
            }
        }
        cur_cols_point_in_counters[i] = counterPosition;
        i++;
        rowBit += width;
    }

    cols_counter_offset_end[x] = counterPosition < end ? (counterPosition + 1) : end;

    cols_counters_recorded[x] = true;
    return;
};
