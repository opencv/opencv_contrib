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
#include "result.hpp"

using zxing::ArrayRef;
using zxing::Ref;
using zxing::Result;
using zxing::ResultPoint;
using zxing::String;

Result::Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints)
    : text_(text), rawBytes_(rawBytes), resultPoints_(resultPoints) {
    charset_ = "UTF-8";

    qrcodeVersion_ = -1;
    pyramidLv_ = -1;
    binaryMethod_ = -1;
    ecLevel_ = '0';
}

Result::Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints,
               std::string charset)
    : text_(text), rawBytes_(rawBytes), resultPoints_(resultPoints), charset_(charset) {
    qrcodeVersion_ = -1;
    pyramidLv_ = -1;
    binaryMethod_ = -1;
    ecLevel_ = '0';
}

Result::Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints,
               std::string charset, int QRCodeVersion, string ecLevel, string charsetMode)
    : text_(text),
      rawBytes_(rawBytes),
      resultPoints_(resultPoints),
      charset_(charset),
      qrcodeVersion_(QRCodeVersion),
      ecLevel_(ecLevel),
      charsetMode_(charsetMode) {
    pyramidLv_ = -1;
    binaryMethod_ = -1;
}

Result::~Result() {}

Ref<String> Result::getText() { return text_; }

ArrayRef<char> Result::getRawBytes() { return rawBytes_; }

ArrayRef<Ref<ResultPoint> > const& Result::getResultPoints() const { return resultPoints_; }

ArrayRef<Ref<ResultPoint> >& Result::getResultPoints() { return resultPoints_; }

void Result::enlargeResultPoints(int scale) {
    for (int i = 0; i < resultPoints_->size(); i++) {
        resultPoints_[i] = Ref<ResultPoint>(new ResultPoint(
            resultPoints_[i]->getX() * (float)scale, resultPoints_[i]->getY() * (float)scale));
    }
    return;
}

std::string Result::getCharset() const { return charset_; }

std::string zxing::Result::getChartsetMode() const { return charsetMode_; }
