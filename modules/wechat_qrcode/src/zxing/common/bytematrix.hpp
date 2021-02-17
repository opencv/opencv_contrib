// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_COMMON_BYTEMATRIX_HPP__
#define __ZXING_COMMON_BYTEMATRIX_HPP__

#include "../errorhandler.hpp"
#include "array.hpp"
#include "bitarray.hpp"
#include "counted.hpp"

namespace zxing {

class ByteMatrix : public Counted {
public:
    explicit ByteMatrix(int dimension);
    ByteMatrix(int _width, int _height);
    ByteMatrix(int _width, int _height, ArrayRef<char> source);
    ~ByteMatrix();

    char get(int x, int y) const {
        int offset = row_offsets[y] + x;
        return bytes[offset];
    }

    void set(int x, int y, char char_value) {
        int offset = row_offsets[y] + x;
        bytes[offset] = char_value & 0XFF;
    }

    unsigned char* getByteRow(int y, ErrorHandler& err_handler);

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    unsigned char* bytes;

private:
    int width;
    int height;

    // ArrayRef<char> bytes;
    // ArrayRef<int> row_offsets;
    int* row_offsets;

private:
    inline void init(int, int);
    ByteMatrix(const ByteMatrix&);
    ByteMatrix& operator=(const ByteMatrix&);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_BYTEMATRIX_HPP__
