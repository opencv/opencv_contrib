// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_DECODER_RESULT_HPP__
#define __ZXING_COMMON_DECODER_RESULT_HPP__

#include "../qrcode/decoder/qrcode_decoder_metadata.hpp"
#include "array.hpp"
#include "counted.hpp"
#include "str.hpp"

namespace zxing {

class DecoderResult : public Counted {
private:
    ArrayRef<char> rawBytes_;
    Ref<String> text_;
    ArrayRef<ArrayRef<char> > byteSegments_;
    std::string ecLevel_;
    std::string outputCharset_;
    int qrcodeVersion_;
    std::string charsetMode_;

    Ref<qrcode::QRCodeDecoderMetaData> other_;
    string otherClassName;

public:
    DecoderResult(ArrayRef<char> rawBytes, Ref<String> text,
                  ArrayRef<ArrayRef<char> >& byteSegments, std::string const& ecLevel);

    DecoderResult(ArrayRef<char> rawBytes, Ref<String> text,
                  ArrayRef<ArrayRef<char> >& byteSegments, std::string const& ecLevel,
                  std::string outputCharset);

    DecoderResult(ArrayRef<char> rawBytes, Ref<String> text,
                  ArrayRef<ArrayRef<char> >& byteSegments, std::string const& ecLevel,
                  std::string outputCharset, int qrcodeVersion, std::string& charsetMode);

    DecoderResult(ArrayRef<char> rawBytes, Ref<String> text);

    DecoderResult(ArrayRef<char> rawBytes, Ref<String> text, std::string outputCharset);

    ArrayRef<char> getRawBytes();
    Ref<String> getText();
    std::string getCharset();

    void setOther(Ref<qrcode::QRCodeDecoderMetaData> other) {
        other_ = other;
        otherClassName = "QRCodeDecoderMetaData";
    };

    Ref<qrcode::QRCodeDecoderMetaData> getOther() {
        // className = otherClassName;
        return other_;
    };

    string getOtherClassName() { return otherClassName; };

    int getQRCodeVersion() const { return qrcodeVersion_; };

    void setText(Ref<String> text) { text_ = text; };

    string getEcLevel() { return ecLevel_; }

    string getCharsetMode() { return charsetMode_; }
};

}  // namespace zxing

#endif  // __ZXING_COMMON_DECODER_RESULT_HPP__
