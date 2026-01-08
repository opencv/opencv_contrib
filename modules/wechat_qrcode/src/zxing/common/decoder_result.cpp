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
#include "decoder_result.hpp"

using zxing::DecoderResult;
using zxing::Ref;
using zxing::ArrayRef;
using zxing::String;
DecoderResult::DecoderResult(ArrayRef<char> rawBytes, Ref<String> text,
                             ArrayRef<ArrayRef<char> >& byteSegments, string const& ecLevel)
    : rawBytes_(rawBytes), text_(text), byteSegments_(byteSegments), ecLevel_(ecLevel) {
    outputCharset_ = "UTF-8";
    otherClassName = "";
    qrcodeVersion_ = -1;
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes, Ref<String> text,
                             ArrayRef<ArrayRef<char> >& byteSegments, string const& ecLevel,
                             string outputCharset)
    : rawBytes_(rawBytes),
      text_(text),
      byteSegments_(byteSegments),
      ecLevel_(ecLevel),
      outputCharset_(outputCharset) {
    otherClassName = "";
    qrcodeVersion_ = -1;
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes, Ref<String> text,
                             ArrayRef<ArrayRef<char> >& byteSegments, string const& ecLevel,
                             string outputCharset, int qrcodeVersion, string& charsetMode)
    : rawBytes_(rawBytes),
      text_(text),
      byteSegments_(byteSegments),
      ecLevel_(ecLevel),
      outputCharset_(outputCharset),
      qrcodeVersion_(qrcodeVersion),
      charsetMode_(charsetMode) {
    otherClassName = "";
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes, Ref<String> text)
    : rawBytes_(rawBytes), text_(text) {
    outputCharset_ = "UTF-8";
    otherClassName = "";
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes, Ref<String> text, std::string outputCharset)
    : rawBytes_(rawBytes), text_(text), outputCharset_(outputCharset) {
    otherClassName = "";
}

ArrayRef<char> DecoderResult::getRawBytes() { return rawBytes_; }

Ref<String> DecoderResult::getText() { return text_; }

string DecoderResult::getCharset() { return outputCharset_; }
