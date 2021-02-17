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
#include "error_correction_level.hpp"
using zxing::ErrorHandler;

namespace zxing {
namespace qrcode {

ErrorCorrectionLevel::ErrorCorrectionLevel(int inOrdinal, int bits, char const* name)
    : ordinal_(inOrdinal), bits_(bits), name_(name) {}

int ErrorCorrectionLevel::ordinal() const { return ordinal_; }

int ErrorCorrectionLevel::bits() const { return bits_; }

string const& ErrorCorrectionLevel::name() const { return name_; }

ErrorCorrectionLevel::operator string const &() const { return name_; }

ErrorCorrectionLevel& ErrorCorrectionLevel::forBits(int bits, ErrorHandler& err_handler) {
    if (bits < 0 || bits >= N_LEVELS) {
        err_handler = zxing::ReaderErrorHandler("Ellegal error correction level bits");
        return *FOR_BITS[0];
    }
    return *FOR_BITS[bits];
}

ErrorCorrectionLevel ErrorCorrectionLevel::L(0, 0x01, "L");
ErrorCorrectionLevel ErrorCorrectionLevel::M(1, 0x00, "M");
ErrorCorrectionLevel ErrorCorrectionLevel::Q(2, 0x03, "Q");
ErrorCorrectionLevel ErrorCorrectionLevel::H(3, 0x02, "H");
ErrorCorrectionLevel* ErrorCorrectionLevel::FOR_BITS[] = {&M, &L, &H, &Q};
int ErrorCorrectionLevel::N_LEVELS = 4;

}  // namespace qrcode
}  // namespace zxing
