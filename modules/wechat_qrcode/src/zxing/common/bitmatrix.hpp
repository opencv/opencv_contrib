// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_BITMATRIX_HPP__
#define __ZXING_COMMON_BITMATRIX_HPP__

#include "../errorhandler.hpp"
#include "array.hpp"
#include "bitarray.hpp"
#include "counted.hpp"
using namespace std;

namespace zxing {

class BitMatrix : public Counted {
public:
    static const int bitsPerWord = std::numeric_limits<unsigned int>::digits;

private:
    int width;
    int height;
    int rowBitsSize;

    vector<COUNTER_TYPE> row_counters;
    vector<COUNTER_TYPE> row_counters_offset;
    vector<bool> row_counters_recorded;
    vector<COUNTER_TYPE> row_counter_offset_end;
    vector<COUNTER_TYPE> row_point_offset;

    vector<COUNTER_TYPE> cols_counters;
    vector<COUNTER_TYPE> cols_counters_offset;
    vector<bool> cols_counters_recorded;
    vector<COUNTER_TYPE> cols_counter_offset_end;
    vector<COUNTER_TYPE> cols_point_offset;

    ArrayRef<unsigned char> bits;
    ArrayRef<int> rowOffsets;

public:
    BitMatrix(int _width, int _height, unsigned char* bitsPtr, ErrorHandler& err_handler);
    BitMatrix(int dimension, ErrorHandler& err_handler);
    BitMatrix(int _width, int _height, ErrorHandler& err_handler);

    void copyOf(Ref<BitMatrix> _bits, ErrorHandler& err_handler);
    void xxor(Ref<BitMatrix> _bits);

    ~BitMatrix();

    unsigned char get(int x, int y) const { return bits[width * y + x]; }

    void set(int x, int y) { bits[rowOffsets[y] + x] = (unsigned char)1; }

    void set(int x, int y, unsigned char value) { bits[rowOffsets[y] + x] = value; }

    void swap(int srcX, int srcY, int dstX, int dstY) {
        auto temp = bits[width * srcY + srcX];
        bits[width * srcY + srcX] = bits[width * dstY + dstX];
        bits[width * dstY + dstX] = temp;
    }

    void getRowBool(int y, bool* row);
    bool* getRowBoolPtr(int y);
    void setRowBool(int y, bool* row);
    int getRowBitsSize() { return rowBitsSize; }
    unsigned char* getPtr() { return bits->data(); }

    void flip(int x, int y);
    void flipAll();
    void clear();
    void setRegion(int left, int top, int _width, int _height, ErrorHandler& err_handler);
    void flipRegion(int left, int top, int _width, int _height, ErrorHandler& err_handler);
    Ref<BitArray> getRow(int y, Ref<BitArray> row);

    int getWidth() const;
    int getHeight() const;

    ArrayRef<int> getTopLeftOnBit() const;
    ArrayRef<int> getBottomRightOnBit() const;

    bool isInitRowCounters;
    void initRowCounters();
    COUNTER_TYPE* getRowRecords(int y);
    COUNTER_TYPE* getRowRecordsOffset(int y);
    bool getRowFirstIsWhite(int y);
    COUNTER_TYPE getRowCounterOffsetEnd(int y);
    bool getRowLastIsWhite(int y);
    COUNTER_TYPE* getRowPointInRecords(int y);

    bool isInitColsCounters;
    void initColsCounters();
    COUNTER_TYPE* getColsRecords(int x);
    COUNTER_TYPE* getColsRecordsOffset(int x);
    COUNTER_TYPE* getColsPointInRecords(int x);
    COUNTER_TYPE getColsCounterOffsetEnd(int x);

private:
    inline void init(int, int, ErrorHandler& err_handler);
    inline void init(int _width, int _height, unsigned char* bitsPtr, ErrorHandler& err_handler);

    void setRowRecords(int y);
    void setColsRecords(int x);

    BitMatrix(const BitMatrix&, ErrorHandler& err_handler);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_BITMATRIX_HPP__
