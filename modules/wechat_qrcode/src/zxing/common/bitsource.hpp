// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_BITSOURCE_HPP__
#define __ZXING_COMMON_BITSOURCE_HPP__

#include "../errorhandler.hpp"
#include "array.hpp"

namespace zxing {
/**
 * <p>This provides an easy abstraction to read bits at a time from a sequence
 * of bytes, where the number of bits read is not often a multiple of 8.</p>
 *
 * <p>This class is not thread-safe.</p>
 *
 * @author srowen@google.com (Sean Owen)
 * @author christian.brunschen@gmail.com (Christian Brunschen)
 */
class BitSource : public Counted {
    typedef char byte;

private:
    ArrayRef<byte> bytes_;
    int byteOffset_;
    int bitOffset_;

public:
    /**
     * @param bytes bytes from which this will read bits. Bits will be read from
     * the first byte first. Bits are read within a byte from most-significant
     * to least-significant bit.
     */
    explicit BitSource(ArrayRef<byte> &bytes) : bytes_(bytes), byteOffset_(0), bitOffset_(0) {}

    int getBitOffset() { return bitOffset_; }

    int getByteOffset() { return byteOffset_; }

    int readBits(int numBits, ErrorHandler &err_handler);

    /**
     * @return number of bits that can be read successfully
     */
    int available();
};

}  // namespace zxing

#endif  // __ZXING_COMMON_BITSOURCE_HPP__
