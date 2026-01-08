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
#include "bitsource.hpp"
#include <sstream>

namespace zxing {

int BitSource::readBits(int numBits, ErrorHandler& err_handler) {
    if (numBits < 0 || numBits > 32 || numBits > available()) {
        std::ostringstream oss;
        oss << numBits;
        err_handler = IllegalArgumentErrorHandler(oss.str().c_str());
        return -1;
    }

    int result = 0;

    // First, read remainder from current byte
    if (bitOffset_ > 0) {
        int bitsLeft = 8 - bitOffset_;
        int toRead = numBits < bitsLeft ? numBits : bitsLeft;
        int bitsToNotRead = bitsLeft - toRead;
        int mask = (0xFF >> (8 - toRead)) << bitsToNotRead;
        result = (bytes_[byteOffset_] & mask) >> bitsToNotRead;
        numBits -= toRead;
        bitOffset_ += toRead;
        if (bitOffset_ == 8) {
            bitOffset_ = 0;
            byteOffset_++;
        }
    }

    // Next read whole bytes
    if (numBits > 0) {
        while (numBits >= 8) {
            result = (result << 8) | (bytes_[byteOffset_] & 0xFF);
            byteOffset_++;
            numBits -= 8;
        }

        // Finally read a partial byte
        if (numBits > 0) {
            int bitsToNotRead = 8 - numBits;
            int mask = (0xFF >> bitsToNotRead) << bitsToNotRead;
            result = (result << numBits) | ((bytes_[byteOffset_] & mask) >> bitsToNotRead);
            bitOffset_ += numBits;
        }
    }

    return result;
}

int BitSource::available() { return 8 * (bytes_->size() - byteOffset_) - bitOffset_; }
}  // namespace zxing
