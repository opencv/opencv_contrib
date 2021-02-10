// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DECODER_MODE_HPP__
#define __ZXING_QRCODE_DECODER_MODE_HPP__

#include "../../common/counted.hpp"
#include "../../errorhandler.hpp"
#include "../version.hpp"

namespace zxing {
namespace qrcode {

class Mode {
private:
    int characterCountBitsForVersions0To9_;
    int characterCountBitsForVersions10To26_;
    int characterCountBitsForVersions27AndHigher_;
    int bits_;
    std::string name_;

    Mode(int cbv0_9, int cbv10_26, int cbv27, int bits, char const* name);

public:
    static Mode TERMINATOR;
    static Mode NUMERIC;
    static Mode ALPHANUMERIC;
    static Mode STRUCTURED_APPEND;
    static Mode BYTE;
    static Mode ECI;
    static Mode KANJI;
    static Mode FNC1_FIRST_POSITION;
    static Mode FNC1_SECOND_POSITION;
    static Mode HANZI;

    static Mode& forBits(int bits, ErrorHandler& err_handler);
    // int getCharacterCountBits(Version *version);
    int getCharacterCountBits(Version* version) const;
    int getBits() const;
    string getName() const;
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_MODE_HPP__
