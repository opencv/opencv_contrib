// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../precomp.hpp"
#include "luminance_source.hpp"
#include <sstream>

using zxing::LuminanceSource;
using zxing::Ref;

LuminanceSource::LuminanceSource(int width, int height)
    : width_(width), height_(height) {}

LuminanceSource::~LuminanceSource() {}

bool LuminanceSource::isCropSupported() const { return false; }

Ref<LuminanceSource> LuminanceSource::crop(int, int, int, int, zxing::ErrorHandler&) const {
    return Ref<LuminanceSource>();
}

bool LuminanceSource::isRotateSupported() const { return false; }

Ref<LuminanceSource> LuminanceSource::rotateCounterClockwise(zxing::ErrorHandler&) const {
    return Ref<LuminanceSource>();
}

LuminanceSource::operator std::string() const {
    ArrayRef<char> row;
    std::ostringstream oss;
    zxing::ErrorHandler err_handler;
    for (int y = 0; y < getHeight(); y++) {
        err_handler.Reset();
        row = getRow(y, row, err_handler);
        if (err_handler.ErrCode()) continue;
        for (int x = 0; x < getWidth(); x++) {
            int luminance = row[x] & 0xFF;
            char c;
            if (luminance < 0x40) {
                c = '#';
            } else if (luminance < 0x80) {
                c = '+';
            } else if (luminance < 0xC0) {
                c = '.';
            } else {
                c = ' ';
            }
            oss << c;
        }
        oss << '\n';
    }
    return oss.str();
}
