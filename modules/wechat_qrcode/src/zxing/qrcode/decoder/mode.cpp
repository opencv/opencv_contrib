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
#include "mode.hpp"
#include "../../common/counted.hpp"
#include "../../zxing.hpp"
#include "../version.hpp"

#include <sstream>

using std::ostringstream;
using zxing::qrcode::Mode;

// VC++
using zxing::ErrorHandler;
using zxing::qrcode::Version;

Mode Mode::TERMINATOR(0, 0, 0, 0x00, "TERMINATOR");
Mode Mode::NUMERIC(10, 12, 14, 0x01, "NUMERIC");
Mode Mode::ALPHANUMERIC(9, 11, 13, 0x02, "ALPHANUMERIC");
Mode Mode::STRUCTURED_APPEND(0, 0, 0, 0x03, "STRUCTURED_APPEND");
Mode Mode::BYTE(8, 16, 16, 0x04, "BYTE");
Mode Mode::ECI(0, 0, 0, 0x07, "ECI");
Mode Mode::KANJI(8, 10, 12, 0x08, "KANJI");
Mode Mode::FNC1_FIRST_POSITION(0, 0, 0, 0x05, "FNC1_FIRST_POSITION");
Mode Mode::FNC1_SECOND_POSITION(0, 0, 0, 0x09, "FNC1_SECOND_POSITION");
Mode Mode::HANZI(8, 10, 12, 0x0D, "HANZI");

// Mode::Mode(int cbv0_9, int cbv10_26, int cbv27, int /* bits */, char const*
// name) :
Mode::Mode(int cbv0_9, int cbv10_26, int cbv27, int bits, char const* name)
    : characterCountBitsForVersions0To9_(cbv0_9),
      characterCountBitsForVersions10To26_(cbv10_26),
      // characterCountBitsForVersions27AndHigher_(cbv27), name_(name) {
      characterCountBitsForVersions27AndHigher_(cbv27),
      bits_(bits),
      name_(name) {}

Mode& Mode::forBits(int bits, ErrorHandler& err_handler) {
    switch (bits) {
        case 0x0:
            return TERMINATOR;
        case 0x1:
            return NUMERIC;
        case 0x2:
            return ALPHANUMERIC;
        case 0x3:
            return STRUCTURED_APPEND;
        case 0x4:
            return BYTE;
        case 0x5:
            return FNC1_FIRST_POSITION;
        case 0x7:
            return ECI;
        case 0x8:
            return KANJI;
        case 0x9:
            return FNC1_SECOND_POSITION;
        case 0xD:
            // 0xD is defined in GBT 18284-2000, may not be supported in foreign
            // country
            return HANZI;
        default:
            ostringstream s;
            s << "Illegal mode bits: " << bits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return TERMINATOR;
    }
}

int Mode::getCharacterCountBits(Version* version) const {
    int number = version->getVersionNumber();
    if (number <= 9) {
        return characterCountBitsForVersions0To9_;
    } else if (number <= 26) {
        return characterCountBitsForVersions10To26_;
    } else {
        return characterCountBitsForVersions27AndHigher_;
    }
}

int Mode::getBits() const { return bits_; }

string zxing::qrcode::Mode::getName() const { return name_; }
